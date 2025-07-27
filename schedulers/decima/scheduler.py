from collections.abc import Iterable
from typing import Any
from torch import Tensor

import torch
import torch.nn as nn
from torch_scatter import segment_csr
import torch_geometric as pyg
import torch_sparse

from ..scheduler import TrainableScheduler
from .env_wrapper import DecimaEnvWrapper
from . import utils


class DecimaScheduler(TrainableScheduler):
    """Original Decima architecture, which uses asynchronous message passing
    as in DAGNN.
    Paper: https://dl.acm.org/doi/abs/10.1145/3341302.3342080
    """

    def __init__(
        self,
        num_executors: int,
        embed_dim: int,
        gnn_mlp_kwargs: dict[str, Any],
        policy_mlp_kwargs: dict[str, Any],
        state_dict_path: str | None = None,
        opt_cls: str | None = None,
        opt_kwargs: dict[str, Any] | None = None,
        max_grad_norm: float | None = None,
        num_node_features: int = 5,
        num_dag_features: int = 3,
        **kwargs,
    ):
        super().__init__()

        self.name = "Decima"
        self.env_wrapper_cls = DecimaEnvWrapper
        self.max_grad_norm = max_grad_norm
        self.num_executors = num_executors

        self.encoder = EncoderNetwork(num_node_features, embed_dim, gnn_mlp_kwargs)

        emb_dims = {"node": embed_dim, "dag": embed_dim, "glob": embed_dim}

        self.stage_policy_network = StagePolicyNetwork(
            num_node_features, emb_dims, policy_mlp_kwargs
        )

        self.exec_policy_network = ExecPolicyNetwork(
            num_executors, num_dag_features, emb_dims, policy_mlp_kwargs
        )

        self._reset_biases()

        if state_dict_path:
            self.name += f":{state_dict_path}"
            self.load_state_dict(torch.load(state_dict_path, map_location='cpu'))  # Load to CPU first, then move to device

        if opt_cls:
            self.optim = getattr(torch.optim, opt_cls)(
                self.parameters(), **(opt_kwargs or {})
            )

    def _reset_biases(self) -> None:
        for name, param in self.named_parameters():
            if "bias" in name:
                param.data.zero_()

    @torch.no_grad()
    def schedule(self, obs: dict) -> tuple[dict, dict]:
        dag_batch = utils.obs_to_pyg(obs) #converts the observation to a PyG batch.
        stage_to_job_map = dag_batch.batch
        stage_mask = dag_batch["stage_mask"]

        dag_batch.to(self.device, non_blocking=True)

        # 1. compute node, dag, and global representations
        h_dict = self.encoder(dag_batch)

        # 2. select a schedulable stage
        stage_scores = self.stage_policy_network(dag_batch, h_dict)
        stage_idx, stage_lgprob = utils.sample(stage_scores)

        # retrieve index of selected stage's job
        stage_idx_glob = pyg.utils.mask_to_index(stage_mask)[stage_idx]
        job_idx = stage_to_job_map[stage_idx_glob].item()

        '''
        stage_mask = [False, True, False, True]  # Only stages 1 and 3 are schedulable
        stage_idx = 0  # The first schedulable stage is selected
        stage_to_job_map = [0, 0, 1, 1]  # Stages 0 and 1 belong to Job 0, Stages 2 and 3 belong to Job 1

        stage_idx_glob = pyg.utils.mask_to_index(stage_mask)[stage_idx]
        # stage_idx_glob = [1, 3][0] = 1

        job_idx = stage_to_job_map[stage_idx_glob].item()
        # job_idx = stage_to_job_map[1] = 0

        job_idx = 0  # The selected stage belongs to Job 0
        '''

        # 3. select the number of executors to add to that stage, conditioned
        # on that stage's job
        exec_scores = self.exec_policy_network(dag_batch, h_dict, job_idx)
        num_exec, exec_lgprob = utils.sample(exec_scores)

        action = {"stage_idx": stage_idx, "job_idx": job_idx, "num_exec": num_exec}

        lgprob = stage_lgprob + exec_lgprob

        return action, {"lgprob": lgprob}

    def evaluate_actions(
        self, obsns: Iterable[dict], actions: Iterable[tuple]
    ) -> dict[str, Tensor]:
        dag_batch = utils.collate_obsns(obsns)
        #combines multiple observation into a single pytorch geometric batch.
        actions_ten = torch.tensor(actions)

        # split columns of `actions` into separate tensors
        # NOTE: columns need to be cloned to avoid in-place operation
        stage_selections, job_indices, exec_selections = [
            col.clone() for col in actions_ten.T
        ]

        num_stage_acts = dag_batch["num_stage_acts"]
        num_exec_acts = dag_batch["num_exec_acts"]
        num_nodes_per_obs = dag_batch["num_nodes_per_obs"]
        obs_ptr = dag_batch["obs_ptr"]
        job_indices += obs_ptr[:-1]

        # re-feed all the observations into the model with grads enabled
        dag_batch.to(self.device)
        h_dict = self.encoder(dag_batch)
        stage_scores = self.stage_policy_network(dag_batch, h_dict)
        exec_scores = self.exec_policy_network(dag_batch, h_dict, job_indices)

        stage_lgprobs, stage_entropies = utils.evaluate(
            stage_scores.cpu(), num_stage_acts, stage_selections
        )

        exec_lgprobs, exec_entropies = utils.evaluate(
            exec_scores.cpu(), num_exec_acts[job_indices], exec_selections
        )

        # aggregate the evaluations for nodes and dags
        action_lgprobs = stage_lgprobs + exec_lgprobs

        action_entropies = stage_entropies + exec_entropies
        action_entropies /= (self.num_executors * num_nodes_per_obs).log()

        return {"lgprobs": action_lgprobs, "entropies": action_entropies}


class EncoderNetwork(nn.Module):
    def __init__(
        self, num_node_features: int, embed_dim: int, mlp_kwargs: dict[str, Any]
    ) -> None:
        super().__init__()

        self.node_encoder = NodeEncoder(num_node_features, embed_dim, mlp_kwargs) #processes nodes
        self.dag_encoder = DagEncoder(num_node_features, embed_dim, mlp_kwargs) #processes dags
        self.global_encoder = GlobalEncoder(embed_dim, mlp_kwargs) #processes global features

    def forward(self, dag_batch: pyg.data.Batch) -> dict[str, Tensor]:
        """
        Returns:
            a dict of representations at three different levels:
            node, dag, and global.
        """
        h_node = self.node_encoder(dag_batch)

        h_dag = self.dag_encoder(h_node, dag_batch)

        if "obs_ptr" in dag_batch:
            # batch of obsns
            obs_ptr = dag_batch["obs_ptr"]
            h_glob = self.global_encoder(h_dag, obs_ptr)
        else:
            # single obs
            h_glob = self.global_encoder(h_dag)

        return {"node": h_node, "dag": h_dag, "glob": h_glob}


class NodeEncoder(nn.Module):
    def __init__(
        self,
        num_node_features: int,
        embed_dim: int,
        mlp_kwargs: dict[str, Any],
        reverse_flow: bool = True,
    ) -> None:
        super().__init__()
        self.reverse_flow = reverse_flow
        self.j, self.i = (1, 0) if reverse_flow else (0, 1)

        self.mlp_prep = utils.make_mlp(
            num_node_features, output_dim=embed_dim, **mlp_kwargs
        ) #this makes the multilayer perceptron that prepares the node features
        self.mlp_msg = utils.make_mlp(embed_dim, output_dim=embed_dim, **mlp_kwargs)
        self.mlp_update = utils.make_mlp(embed_dim, output_dim=embed_dim, **mlp_kwargs)

    def forward(self, dag_batch: pyg.data.Batch) -> Tensor:
        """returns a tensor of shape [num_nodes, embed_dim]"""

        edge_masks = dag_batch["edge_masks"]

        if edge_masks.shape[0] == 0:
            # no message passing to do
            return self._forward_no_mp(dag_batch.x)

        # pre-process the node features into initial representations
        h_init = self.mlp_prep(dag_batch.x)

        # will store all the nodes' representations
        h = torch.zeros_like(h_init)

        num_nodes = h.shape[0]

        src_node_mask = ~pyg.utils.index_to_mask(
            dag_batch.edge_index[self.i], num_nodes
        ) # creates a boolean mask for the source nodes. nodes that have no outgoing edges if reverse_flow is True, or no incoming edges if reverse_flow is False

        h[src_node_mask] = self.mlp_update(h_init[src_node_mask]) # initializes the source nodes' representations with the update MLP

        edge_masks_it = (
            iter(reversed(edge_masks)) if self.reverse_flow else iter(edge_masks) 
        )

        # target-to-source message passing, one level of the dags at a time
        for edge_mask in edge_masks_it:
            edge_index_masked = dag_batch.edge_index[:, edge_mask] #boolean mask for the edges that are active in this layer.
            adj = utils.make_adj(edge_index_masked, num_nodes)
            # creates a adjacency matrix.

            # nodes sending messages
            src_mask = pyg.utils.index_to_mask(edge_index_masked[self.j], num_nodes)
            # the shape of edjge_index_masked is [2, num_edges], so we use self.j to get the source nodes and self.i to get the destination nodes which are determined by reverse_flow.
            # src_mask gets us the one from where the messages are sent, and dst_mask gets us the ones where the messages are received.

            # nodes receiving messages
            dst_mask = pyg.utils.index_to_mask(edge_index_masked[self.i], num_nodes)

            msg = torch.zeros_like(h)
            msg[src_mask] = self.mlp_msg(h[src_mask])
            agg = torch_sparse.matmul(adj if self.reverse_flow else adj.t(), msg)
            h[dst_mask] = h_init[dst_mask] + self.mlp_update(agg[dst_mask])


            '''
            # src mask: Boolean mask of shape [num_nodes] where True indicates sender nodes
            # msg : zero-initialized tensor of shape [num_nodes, embed_dim]
            # adj is of the shape [num_nodes, num_nodes].
            # agg : aggregated messages for each node, shape [num_nodes, embed_dim]
            h : zero-initialized tensor of shape [num_nodes, embed_dim].
            '''

        return h

    def _forward_no_mp(self, x: Tensor) -> Tensor:
        """forward pass without any message passing. Needed whenever
        all the active jobs are almost complete and only have a single
        layer of nodes remaining.
        """
        return self.mlp_prep(x)


class DagEncoder(nn.Module):
    def __init__(
        self, num_node_features: int, embed_dim: int, mlp_kwargs: dict[str, Any]
    ) -> None:
        super().__init__()
        input_dim = num_node_features + embed_dim
        self.mlp = utils.make_mlp(input_dim, output_dim=embed_dim, **mlp_kwargs)
        '''
        dag_batch.x is of shape [num_nodes, num_node_features]
        h_node is of shape [num_nodes, embed_dim]
        The input to the MLP is of shape [num_nodes, num_node_features + embed_dim]
        The output of the MLP is of shape [num_dags, embed_dim]
        torch.cat(..) is of shape [num_nodes, num_node_features + embed_dim]
        segment_csr(..) is of shape [num_dags, embed_dim]
        the segment_csr fucntion aggregates the node embeddings for a job or DAG and this has one output per DAG so its of shape [num_dags, embed_dim].
        '''

    def forward(self, h_node: Tensor, dag_batch: pyg.data.Batch) -> Tensor:
        """returns a tensor of shape [num_dags, embed_dim]"""
        # include skip connection from raw input
        h_node = torch.cat([dag_batch.x, h_node], dim=1)
        h_dag = segment_csr(self.mlp(h_node), dag_batch.ptr)
        return h_dag


class GlobalEncoder(nn.Module):
    def __init__(self, embed_dim: int, mlp_kwargs: dict[str, Any]) -> None:
        super().__init__()
        self.mlp = utils.make_mlp(embed_dim, output_dim=embed_dim, **mlp_kwargs)

    def forward(self, h_dag: Tensor, obs_ptr: Tensor | None = None) -> Tensor:
        """returns a tensor of shape [num_observations, embed_dim]"""
        h_dag = self.mlp(h_dag)
        # passes each DAG embedding through an MLP to further refine it.
        # h_dag is of shape [num_nodes, embed_dim] and obs_ptr is of shape [num_observations + 1]
        # h_glob will be of shape [num_observations, embed_dim] after aggregation.

        if obs_ptr is not None:
            # batch of observations
            # obs_ptr is a tensor of shape [num_observations + 1] that contains the start and end indices of each observation in the h_dag tensor.
            # segment_csr will aggregate the DAG embeddings for each observation.
            h_glob = segment_csr(h_dag, obs_ptr)
        else:
            # single observation
            h_glob = h_dag.sum(0).unsqueeze(0)

        return h_glob


class StagePolicyNetwork(nn.Module):
    def __init__(
        self,
        num_node_features: int,
        emb_dims: dict[str, int],
        mlp_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        input_dim = (
            num_node_features + emb_dims["node"] + emb_dims["dag"] + emb_dims["glob"]
        )
        # emb_dims contains the dimensions of the embeddings for node, dag, and global features.
        # eg. emb_dims['node] is The dimension of the node-level embeddings (output of the NodeEncoder for each node).
        self.mlp_score = utils.make_mlp(input_dim, output_dim=1, **mlp_kwargs)

    def forward(self, dag_batch: pyg.data.Batch, h_dict: dict[str, Tensor]) -> Tensor:
        """returns a tensor of shape [num_nodes,]"""

        stage_mask = dag_batch["stage_mask"]
        # stage_mask is a boolean mask of shape [num_nodes,] that indicates which nodes are currently in a stage that can be scheduled.

        x = dag_batch.x[stage_mask]

        h_node = h_dict["node"][stage_mask]
        # h_node is of shape [num_schedulable_nodes, embed_dim]

        batch_masked = dag_batch.batch[stage_mask]
        # dag_batch.batch maps each node to its corresponding job or DAG in the batch.
        # batch_masked is a tensor of shape [num_schedulable_nodes,] that contains the batch indices for the nodes that are currently in a stage that can be scheduled.
        h_dag_rpt = h_dict["dag"][batch_masked]
        # h_dict["dag"] is a tensor of shape [num_dags, embed_dim] that has the DAG embeddings.
        # h_dag_rpt is a tensor of shape [num_schedulable_nodes, embed_dim] that contains the DAG embeddings for the nodes that are currently in a stage that can be scheduled.




        if "num_stage_acts" in dag_batch:
            # batch of obsns
            num_stage_acts = dag_batch["num_stage_acts"]
            # num_stage_acts tells if the current observation is a batch of observation.
            # dag_batch["num_stage_acts"] is a tensor of shape [num_observations,] where each element indicates the number of schedulable nodes in that observation..
        else:
            # single obs
            num_stage_acts = stage_mask.sum()
            # if the current observation is a single observation, then num_stage_acts is the number of schedulable nodes in that observation..



        h_glob_rpt = h_dict["glob"].repeat_interleave(
            num_stage_acts, output_size=h_node.shape[0], dim=0
        )
        '''
        h_dict["glob"] = [[g1], [g2], [g3]]  # Shape: [3, embed_dim]
        num_stage_acts = [2, 1, 3]  # 2 schedulable nodes in obs 1, 1 in obs 2, 3 in obs 3
        h_glob_rpt = [[g1], [g1], [g2], [g3], [g3], [g3]]  # Shape: [6, embed_dim].
        '''



        # residual connections to original features
        node_inputs = torch.cat([x, h_node, h_dag_rpt, h_glob_rpt], dim=1)

        node_scores = self.mlp_score(node_inputs).squeeze(-1)
        return node_scores


class ExecPolicyNetwork(nn.Module):
    def __init__(
        self,
        num_executors: int,
        num_dag_features: int,
        emb_dims: dict[str, int],
        mlp_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        self.num_executors = num_executors
        self.num_dag_features = num_dag_features
        input_dim = num_dag_features + emb_dims["dag"] + emb_dims["glob"] + 1

        self.mlp_score = utils.make_mlp(input_dim, output_dim=1, **mlp_kwargs)

    def forward(
        self, dag_batch: pyg.data.Batch, h_dict: dict[str, Tensor], job_indices: Tensor
    ) -> Tensor:
        exec_mask = dag_batch["exec_mask"]
        '''
        dag_batch["exec_mask"] = [
        [True, True, False, False],  # Job 1: Only 1 or 2 executors are valid
        [True, True, True, False],   # Job 2: Only 1, 2, or 3 executors are valid
        ]
        '''

        dag_start_idxs = dag_batch.ptr[:-1]
        x_dag = dag_batch.x[dag_start_idxs, : self.num_dag_features]
        x_dag = x_dag[job_indices]

        '''
        dag_batch.x = [
        [1.0, 2.0, 3.0],  # Node 0 (DAG 1)
        [4.0, 5.0, 6.0],  # Node 1 (DAG 1)
        [7.0, 8.0, 9.0],  # Node 2 (DAG 2)
        [10.0, 11.0, 12.0],  # Node 3 (DAG 2)
        ]
        dag_batch.ptr = [0, 2, 4]  # DAG 1 starts at 0, DAG 2 starts at 2
        self.num_dag_features = 2  # Only use the first 2 features

        x_dag = dag_batch.x[dag_start_idxs, : self.num_dag_features]
        # dag_start_idxs = [0, 2]
        # x_dag = [
        #     [1.0, 2.0],  # First 2 features of Node 0 (DAG 1)
        #     [7.0, 8.0],  # First 2 features of Node 2 (DAG 2)
        #     ]
        '''

        h_dag = h_dict["dag"][job_indices]

        exec_mask = exec_mask[job_indices]

        if "num_exec_acts" in dag_batch:
            # batch of obsns
            num_exec_acts = dag_batch["num_exec_acts"][job_indices]
        else:
            # single obs
            num_exec_acts = exec_mask.sum()
            x_dag = x_dag.unsqueeze(0)
            h_dag = h_dag.unsqueeze(0)
            exec_mask = exec_mask.unsqueeze(0)

        exec_actions = self._get_exec_actions(exec_mask)
        #exec_actions is of shape [num_exec_acts, 1] and it normalizes the  

        # residual connections to original features
        x_h_dag = torch.cat([x_dag, h_dag], dim=1)

        x_h_dag_rpt = x_h_dag.repeat_interleave(
            num_exec_acts, output_size=exec_actions.shape[0], dim=0
        )

        h_glob_rpt = h_dict["glob"].repeat_interleave(
            num_exec_acts, output_size=exec_actions.shape[0], dim=0
        )

        dag_inputs = torch.cat([x_h_dag_rpt, h_glob_rpt, exec_actions], dim=1)

        dag_scores = self.mlp_score(dag_inputs).squeeze(-1)
        return dag_scores

    def _get_exec_actions(self, exec_mask: Tensor) -> Tensor:
        exec_actions = torch.arange(self.num_executors) / self.num_executors
        exec_actions = exec_actions.to(exec_mask.device)
        exec_actions = exec_actions.repeat(exec_mask.shape[0])
        exec_actions = exec_actions[exec_mask.view(-1)]
        return exec_actions.unsqueeze(-1)
