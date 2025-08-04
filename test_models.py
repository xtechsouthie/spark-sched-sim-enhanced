import os.path as osp
import torch
import gymnasium as gym
from cfg_loader import load
from schedulers import make_scheduler
from spark_sched_sim.wrappers import StochasticTimeLimit
from spark_sched_sim import metrics
import numpy as np
import random

ENV_CFG = {
    "num_executors": 10,
    "job_arrival_cap": 50,
    "job_arrival_rate": 4.0e-5,
    "moving_delay": 2000.0,
    "warmup_delay": 1000.0,
    "data_sampler_cls": "TPCHDataSampler",
    "mean_time_limit": 100000.0,
}

def main():
    cfg = load('test/test.yaml')
    train_cfg, agent_cfg, env_cfg = cfg['trainer'], cfg['agent'], cfg['env']
    num_episodes = 10
    results = {}
    print('\n------Evaluating all models now-------')

    print('------Evaluationg original model-----------')

    original_cfg = load(filename=osp.join("config", "decima_tpch.yaml"))

    original_agent_cfg = original_cfg["agent"] | {
        "num_executors": ENV_CFG["num_executors"],
        "state_dict_path": osp.join("models", "decima", "model.pt"),
    }

    scheduler_original = make_scheduler(original_agent_cfg)

    print("Example: Decima")

    print("Running episode...")
    results['original_model'] = _evaluate_single_model(scheduler_original)


    print('------Evaluating model trained WITHOUT critic-------')

    model_path = "test/artifacts/no_meta_critic/checkpoints/final_no_meta_critic.pt"
    if not osp.exists(model_path):
        print(f"Model not found: {model_path}")
        return None
    
    agent_cfg = cfg["agent"] | {"num_executors": ENV_CFG["num_executors"]}
    
    scheduler_without_critic = make_scheduler(agent_cfg)
    scheduler_without_critic.load_state_dict(torch.load(model_path, map_location='cpu'))
    scheduler_without_critic.eval()
    results['without_critic'] = _evaluate_single_model(scheduler_without_critic)


    print('--------Evaluating model trained WITH critic---------')
    model_path = "test/artifacts/with_meta_critic/checkpoints/final_with_meta_critic.pt"
    if not osp.exists(model_path):
        print(f"Model not found: {model_path}")
        return None
    
    agent_cfg = cfg["agent"] | {"num_executors": ENV_CFG["num_executors"]}
    
    scheduler_with_critic = make_scheduler(agent_cfg)
    scheduler_with_critic.load_state_dict(torch.load(model_path, map_location='cpu'))
    scheduler_with_critic.eval()

    results['with_critic'] = _evaluate_single_model(scheduler_with_critic)

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
    print(f"ORIGINAL job durations: {results['original_model']['job_durations']}")
    print('\n ------------------------------------------')

    return results

# def test_original_model():
#     """Test the original pre-trained model"""
#     print("Testing Original Model...")
    
#     cfg = load(filename=osp.join("config", "decima_tpch.yaml"))
#     agent_cfg = cfg["agent"] | {
#         "num_executors": ENV_CFG["num_executors"],
#         "state_dict_path": osp.join("models", "decima", "model.pt"),
#     }
    
#     scheduler = make_scheduler(agent_cfg)
#     stats = run_episode(scheduler)
#     print(f"Original Model - Avg Job Duration: {stats['avg_job_duration']:.3f}s\n")
#     return stats['avg_job_duration']

# def test_no_critic_model():
#     """Test the model trained without critic"""
#     print("Testing No-Critic Model...")
    
#     model_path = "test/artifacts/no_critic/checkpoints/final_no_critic.pt"
#     if not osp.exists(model_path):
#         print(f"Model not found: {model_path}")
#         return None
    
#     cfg = load(filename=osp.join("config", "decima_tpch.yaml"))
#     agent_cfg = cfg["agent"] | {"num_executors": ENV_CFG["num_executors"]}
    
#     scheduler = make_scheduler(agent_cfg)
#     scheduler.load_state_dict(torch.load(model_path, map_location='cpu'))
#     scheduler.eval()
    
#     avg_duration = run_episode(scheduler)
#     print(f"No-Critic Model - Avg Job Duration: {avg_duration:.3f}s\n")
#     return avg_duration

# def test_critic_model():
#     """Test the model trained with critic"""
#     print("Testing With-Critic Model...")
    
#     model_path = "test/artifacts/with_critic/checkpoints/final_with_critic.pt"
#     if not osp.exists(model_path):
#         print(f"Model not found: {model_path}")
#         return None
    
#     cfg = load(filename=osp.join("config", "decima_tpch.yaml"))
#     agent_cfg = cfg["agent"] | {"num_executors": ENV_CFG["num_executors"]}
    
#     scheduler = make_scheduler(agent_cfg)
#     scheduler.load_state_dict(torch.load(model_path, map_location='cpu'))
#     scheduler.eval()
    
#     avg_duration = run_episode(scheduler)
#     print(f"With-Critic Model - Avg Job Duration: {avg_duration:.3f}s\n")
#     return avg_duration

def _evaluate_single_model(scheduler, env_cfg: dict = ENV_CFG, num_episodes: int = 10, seed=9999) -> dict:
        import gymnasium as gym
        print(f'ENV_CFG: {env_cfg}')
        print(f'\nnum episodes: {num_episodes}')
        print('\n evaluating a model')

        eval_env = gym.make('spark_sched_sim:SparkSchedSimEnv-v0', env_cfg=env_cfg)

        if scheduler.env_wrapper_cls:
            eval_env = scheduler.env_wrapper_cls(eval_env)

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        rewards, job_durations = [], []

        for episode in range(num_episodes):
            obs, _ = eval_env.reset(seed=(seed + episode), options=None) #for getting different obs each time
            print(f'running episode with seed: {seed + episode}')
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
    

if __name__ == "__main__":
    main()