from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any
import shutil
import os
import os.path as osp
import sys
from copy import deepcopy
import json
import pathlib
import random
from cfg_loader import load

import numpy as np
import torch
import multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from spark_sched_sim import metrics

from schedulers import make_scheduler, TrainableScheduler
from .rollout_worker import RolloutWorkerSync, RolloutWorkerAsync, RolloutBuffer
from .utils import Baseline, ReturnsCalculator


CfgType = dict[str, Any]

ENV_CFG = {
    "num_executors": 50,
    "job_arrival_cap": 50,
    "job_arrival_rate": 4.0e-5,
    "moving_delay": 2000.0,
    "warmup_delay": 1000.0,
    "data_sampler_cls": "TPCHDataSampler",
    "render_mode": "human",
}


class Trainer(ABC):
    """Base training algorithm class. Each algorithm must implement the
    abstract method `train_on_rollouts`
    """

    def __init__(
        self, agent_cfg: CfgType, env_cfg: CfgType, train_cfg: CfgType
    ) -> None:
        self.seed = train_cfg["seed"]
        torch.manual_seed(self.seed)

        self.scheduler_cls = agent_cfg["agent_cls"]

        self.device = torch.device(
            train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )

        # number of training iterations
        self.num_iterations: int = train_cfg["num_iterations"]

        # number of unique job sequences per iteration
        self.num_sequences: int = train_cfg["num_sequences"]

        # number of rollouts per job sequence
        self.num_rollouts: int = int(train_cfg["num_rollouts"])
        if self.num_rollouts % 2 != 0:
            raise ValueError(f'num_rollouts must be even, got {self.num_rollouts}')
            # needed to distribute into 2 for meta critic training.

        self.artifacts_dir: str = train_cfg["artifacts_dir"]
        pathlib.Path(self.artifacts_dir).mkdir(parents=True, exist_ok=True)

        self.stdout_dir = osp.join(self.artifacts_dir, "stdout")
        self.tb_dir = osp.join(self.artifacts_dir, "tb")
        self.checkpointing_dir = osp.join(self.artifacts_dir, "checkpoints")
        self.use_tensorboard: bool = train_cfg["use_tensorboard"]
        self.checkpointing_freq: int = train_cfg["checkpointing_freq"]
        self.env_cfg = env_cfg

        self.baseline = Baseline(self.num_sequences, self.num_rollouts)

        self.rollout_duration: float | None = train_cfg.get("rollout_duration")

        assert ("reward_buff_cap" in train_cfg) ^ (
            "beta_discount" in train_cfg
        ), "must provide exactly one of `reward_buff_cap` and `beta_discount` in config"

        if "reward_buff_cap" in train_cfg:
            self.return_calc = ReturnsCalculator(buff_cap=train_cfg["reward_buff_cap"])
        else:
            beta: float = train_cfg["beta_discount"]
            env_cfg |= {"beta": beta}
            self.return_calc = ReturnsCalculator(beta=beta)

        self.scheduler_cfg = (
            agent_cfg
            | {"num_executors": env_cfg["num_executors"]}
            | {k: train_cfg[k] for k in ["opt_cls", "opt_kwargs", "max_grad_norm"]}
        )
        scheduler = make_scheduler(self.scheduler_cfg)
        assert isinstance(scheduler, TrainableScheduler), "scheduler must be trainable."
        self.scheduler: TrainableScheduler = scheduler

    def train(self) -> None:
        """trains the model on different job arrival sequences.
        For each job sequence:
        - multiple rollouts are collected in parallel, asynchronously
        - the rollouts are gathered at the center, where model parameters are
            updated, and
        - new model parameters are scattered to the rollout workers
        """
        self._setup()

        # every n'th iteration, save the best model from the past n iterations,
        # where `n = self.model_save_freq`
        best_state = None

        exception: Exception | None = None

        print("Beginning training.\n", flush=True)

        for i in range(self.num_iterations):
            state_dict = deepcopy(self.scheduler.state_dict())

            # # move params to GPU for learning
            self.scheduler.to(self.device, non_blocking=True)

            # scatter
            for conn in self.conns:
                conn.send({"state_dict": state_dict})

            # gather
            results = []
            for worker_idx, conn in enumerate(self.conns):
                res = conn.recv()
                if isinstance(res, Exception):
                    print(f"An exception occured in process {worker_idx}", flush=True)
                    exception = res
                    break
                results += [res]

            if exception:
                break

            rollout_buffers, rollout_stats_list = zip(
                *[(res["rollout_buffer"], res["stats"]) for res in results if res]
            )

            # update parameters
            learning_stats = self.train_on_rollouts(rollout_buffers)

            # return params to CPU before scattering updated state dict to the rollout workers
            self.scheduler.to("cpu", non_blocking=True)

            avg_num_jobs = self.return_calc.avg_num_jobs or np.mean(
                [stats["avg_num_jobs"] for stats in rollout_stats_list]
            )

            # check if model is the current best
            if not best_state or avg_num_jobs < best_state["avg_num_jobs"]:
                best_state = self._capture_state(
                    i, avg_num_jobs, state_dict, rollout_stats_list
                )

            if (i + 1) % self.checkpointing_freq == 0:
                self._checkpoint(i, best_state)
                best_state = None

            if self.use_tensorboard:
                ep_lens = [len(buff) for buff in rollout_buffers if buff]
                self._write_stats(i, learning_stats, rollout_stats_list, ep_lens)

            print(
                f"Iteration {i+1} complete. Avg. # jobs: " f"{avg_num_jobs:.3f}",
                flush=True,
            )

        self._cleanup()

        if exception:
            raise exception

    @abstractmethod
    def train_on_rollouts(
        self, rollout_buffers: Iterable[RolloutBuffer]
    ) -> dict[str, Any]:
        pass

    # internal methods

    def _preprocess_rollouts(
        self, rollout_buffers: Iterable[RolloutBuffer]
    ) -> dict[str, tuple]:
        (
            obsns_list,
            actions_list,
            wall_times_list,
            rewards_list,
            lgprobs_list,
            resets_list,
        ) = zip(
            *(
                (
                    buff.obsns,
                    buff.actions,
                    buff.wall_times,
                    buff.rewards,
                    buff.lgprobs,
                    buff.resets,
                )
                for buff in rollout_buffers
                if buff is not None
            )
        )

        returns_list = self.return_calc(
            rewards_list,
            wall_times_list,
            resets_list,
        )

        wall_times_list = tuple([wall_times[:-1] for wall_times in wall_times_list])
        baselines_list = self.baseline(wall_times_list, returns_list)

        return {
            "obsns_list": obsns_list,
            "actions_list": actions_list,
            "returns_list": returns_list,
            "baselines_list": baselines_list,
            "lgprobs_list": lgprobs_list,
        }

    def _setup(self) -> None:
        # logging
        shutil.rmtree(self.stdout_dir, ignore_errors=True)
        os.mkdir(self.stdout_dir)
        #sys.stdout = open(osp.join(self.stdout_dir, "main.out"), "a")

        if self.use_tensorboard:
            self.summary_writer = SummaryWriter(self.tb_dir)

        # model checkpoints
        shutil.rmtree(self.checkpointing_dir, ignore_errors=True)
        os.mkdir(self.checkpointing_dir)

        # torch
        try:
            torch.multiprocessing.set_start_method("spawn")
        except RuntimeError as e:
            if "context has already been set" in str(e):
                print("DEBUG: Multiprocessing context already set, skipping", flush=True)
            else:
                raise e
        # print('cuda available:', torch.cuda.is_available())
        # torch.autograd.set_detect_anomaly(True)

        self.scheduler.train()

        self._start_rollout_workers()

    def _cleanup(self) -> None:
        self._terminate_rollout_workers()

        if self.use_tensorboard:
            self.summary_writer.close()

        print("\nTraining complete.", flush=True)

    def _capture_state(
        self, i: int, avg_num_jobs: float, state_dict: dict, stats_list: Iterable[dict]
    ) -> dict[str, Any]:
        return {
            "iteration": i,
            "avg_num_jobs": np.round(avg_num_jobs, 3),
            "state_dict": state_dict,
            "completed_job_count": int(
                np.mean([stats["num_completed_jobs"] for stats in stats_list])
            ),
        }

    def _checkpoint(self, i: int, best_state: dict) -> None:
        dir = osp.join(self.checkpointing_dir, f"{i+1}")
        os.mkdir(dir)
        best_sd = best_state.pop("state_dict")
        torch.save(best_sd, osp.join(dir, "model.pt"))
        with open(osp.join(dir, "state.json"), "w") as fp:
            json.dump(best_state, fp)

    def _start_rollout_workers(self) -> None:
        self.procs = []
        self.conns = []

        base_seeds = self.seed + np.arange(self.num_sequences)
        base_seeds = np.repeat(base_seeds, self.num_rollouts)
        seed_step = self.num_sequences
        lock = mp.Lock()
        for rank, base_seed in enumerate(base_seeds):
            conn_main, conn_sub = mp.Pipe()
            self.conns += [conn_main]

            proc = mp.Process(
                target=RolloutWorkerAsync(self.rollout_duration)
                if self.rollout_duration
                else RolloutWorkerSync(),
                args=(
                    rank,
                    conn_sub,
                    self.env_cfg,
                    self.scheduler_cfg,
                    self.stdout_dir,
                    int(base_seed),
                    seed_step,
                    lock,
                ),
            )

            self.procs += [proc]
            proc.start()

        for proc in self.procs:
            proc.join(5)

    def _terminate_rollout_workers(self) -> None:
        for conn in self.conns:
            conn.send(None)

        for proc in self.procs:
            proc.join()

    def _write_stats(
        self,
        epoch: int,
        learning_stats: dict,
        stats_list: Iterable[dict],
        ep_lens: list[int],
    ) -> None:
        episode_stats = learning_stats | {
            "avg num concurrent jobs": np.mean(
                [stats["avg_num_jobs"] for stats in stats_list]
            ),
            "avg job duration": np.mean(
                [stats["avg_job_duration"] for stats in stats_list]
            ),
            "completed jobs count": np.mean(
                [stats["num_completed_jobs"] for stats in stats_list]
            ),
            "job arrival count": np.mean(
                [stats["num_job_arrivals"] for stats in stats_list]
            ),
            "episode length": np.mean(ep_lens),
        }

        for name, stat in episode_stats.items():
            self.summary_writer.add_scalar(name, stat, epoch)


    def train_and_compare_models(self, train_cfg: dict, agent_cfg: dict, env_cfg: dict) -> dict:
        print('Training and comparing two models: with and without meta critic')

        original_use_critic = train_cfg.get('use_meta_critic', False)
        original_artifacts_dir = train_cfg.get('artifacts_dir', 'artifacts')
        num_episodes = train_cfg.get('num_episodes')
        from trainers import make_trainer

        results = {}

        print('\n------Training model without critic------')
        cfg_without_critic = {
            'trainer': train_cfg.copy(),
            'agent': agent_cfg.copy(),
            'env': env_cfg.copy()
        }
        cfg_without_critic['trainer']['use_meta_critic'] = False
        cfg_without_critic['trainer']['artifacts_dir'] = f'{original_artifacts_dir}/no_meta_critic'

        trainer_without_critic = make_trainer(cfg_without_critic)
        trainer_without_critic.train()

        without_critic_model_path = f'{cfg_without_critic["trainer"]["artifacts_dir"]}/checkpoints/final_no_meta_critic.pt'
        trainer_without_critic.save(without_critic_model_path)

        print('\n------Training model with critic-------')
        cfg_with_critic = {
            'trainer': train_cfg.copy(),
            'agent': agent_cfg.copy(),
            'env': env_cfg.copy()
        }
        cfg_with_critic['trainer']['use_meta_critic'] = True
        cfg_with_critic['trainer']['artifacts_dir'] = f'{original_artifacts_dir}/with_meta_critic'

        trainer_with_critic = make_trainer(cfg_with_critic)
        trainer_with_critic.train()

        with_critic_model_path = f'{cfg_with_critic["trainer"]["artifacts_dir"]}/checkpoints/final_with_meta_critic.pt'
        trainer_with_critic.save(with_critic_model_path)

        print('\n------Evaluating both models now-------')

        print('------Evaluationg original model-----------')

        original_cfg = load(filename=osp.join("config", "decima_tpch.yaml"))

        original_agent_cfg = original_cfg["agent"] | {
            "num_executors": ENV_CFG["num_executors"],
            "state_dict_path": osp.join("models", "decima", "model.pt"),
        }

        scheduler_original = make_scheduler(original_agent_cfg)

        print("Example: Decima")

        print("Running episode...")
        results['original_model'] = self._evaluate_single_model(scheduler_original, env_cfg, num_episodes)


        print('------Evaluating model trained WITHOUT critic-------')
        # Need to include num_executors from env_cfg for scheduler creation
        eval_agent_cfg = agent_cfg.copy()
        eval_agent_cfg['num_executors'] = env_cfg['num_executors']
        
        scheduler_without_critic = make_scheduler(eval_agent_cfg)
        scheduler_without_critic.load_state_dict(torch.load(without_critic_model_path, map_location=self.device))
        scheduler_without_critic.eval()

        results['without_critic'] = self._evaluate_single_model(scheduler_without_critic, env_cfg, num_episodes)

        print('--------Evaluating model trained WITH critic---------')
        scheduler_with_critic = make_scheduler(eval_agent_cfg)
        scheduler_with_critic.load_state_dict(torch.load(with_critic_model_path, map_location=self.device))
        scheduler_with_critic.eval()

        results['with_critic'] = self._evaluate_single_model(scheduler_with_critic, env_cfg, num_episodes)

        #Comparing the results
        original_model_duration = results["original_model"]['avg_job_duration']
        without_critic_duration = results["without_critic"]['avg_job_duration']
        with_critic_duration = results["with_critic"]['avg_job_duration']
        critic_improvement_over_original = ((original_model_duration - with_critic_duration)/ original_model_duration) * 100
        no_critic_improvement_over_original = ((original_model_duration - without_critic_duration)/original_model_duration) * 100
        improvement = ((without_critic_duration - with_critic_duration) / without_critic_duration) * 100

        results['comparison'] = {
            'without_critic_avg_duration': without_critic_duration,
            'with_critic_avg_duration': with_critic_duration,
            'absolute_improvement': without_critic_duration - with_critic_duration,
            'percentage_improvement': improvement,
            'critic_improvement_over_original': critic_improvement_over_original,
            'no_critic_improvement_over_original': no_critic_improvement_over_original,
            'better_method_bw_critic_baseline': 'CRITIC' if improvement > 0 else 'BASELINE',
            'better_method_bw_critic_original': 'CRITIC' if critic_improvement_over_original > 0 else 'ORIGINAL',
        }

        print('\n -----------COMPARISON RESULTS-----------')
        print(f"\nNo critic model avg job duration: {without_critic_duration:.3f}s")
        print(f"With critic model avg job duration: {with_critic_duration:.3f}s")
        print(f"original model avg job duration: {original_model_duration:.3f}s")
        print(f"\nImprovement of CRITIC over BASELINE: {improvement:.2f}%")
        print(f"Improvement of CRITIC over ORIGINAL: {critic_improvement_over_original:.2f}%")
        print(f"Improvement of BASELINE over ORIGINAL: {no_critic_improvement_over_original:.2f}%")
        print(f"Better Method between CRITIC and BASELINE: {results['comparison']['better_method_bw_critic_baseline']}")
        print(f"Better method between CRITIC and ORIGINAL: {results['comparison']['better_method_bw_critic_original']}")
        print(f"\nCRITIC job durations: {results['with_critic']['job_durations']}")
        print(f"BASELINE job durations: {results['without_critic']['job_durations']}")
        print(f"CRITIC job durations: {results['original_model']['job_durations']}")
        print('\n ------------------------------------------')

        return results
    
    def _evaluate_single_model(self, scheduler, env_cfg: dict, num_episodes: int = 10) -> dict:
        import gymnasium as gym
        import spark_sched_sim  # Import to register the environment

        eval_env = gym.make('spark_sched_sim:SparkSchedSimEnv-v0', env_cfg=env_cfg)

        if scheduler.env_wrapper_cls:
            eval_env = scheduler.env_wrapper_cls(eval_env)

        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        rewards, job_durations = [], []

        for episode in range(num_episodes):
            obs, _ = eval_env.reset(seed=(self.seed + episode), options=None) #for getting different obs each time
            print(f'seed: {self.seed + episode}')
            done = False
            ep_reward = 0

            while not done:
                action, _ = scheduler.schedule(obs)
                obs, reward, term, trunc, info = eval_env.step(action)
                # here the reward is -(sum of the ages of all active jobs)
                done = term or trunc
                ep_reward += reward

            avg_job_duration = metrics.avg_job_duration(eval_env) * 1e-3
            job_durations.append(avg_job_duration)
            rewards.append(ep_reward)

        eval_env.close()

        return {
            'avg_job_duration': np.mean(job_durations),
            'std_job_duration': np.std(job_durations),
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'job_durations': job_durations,
            'rewards': rewards
        }
