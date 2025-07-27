import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from schedulers.decima import utils

class CriticNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim1: int = 128, hidden_dim2: int = 64):
        super(CriticNetwork, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x).squeeze(-1)
    
class Critic:
    def __init__(self, input_dim: int, hidden_dim1: int = 256, hidden_dim2: int = 64,
                 lr: float = 3e-4, num_sequences: int = 1, num_rollouts: int = 1,
                 feature_extractor = None):
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.lr = lr
        self.num_sequences = num_sequences
        self.num_rollouts = num_rollouts
        self.feature_extractor = feature_extractor

        self.network = CriticNetwork(input_dim, hidden_dim1, hidden_dim2)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #self.device nahi hai
        self.network.to(self.device)

        self.state_mean = None
        self.state_std = None
        self.returns_mean = None
        self.returns_std = None
        self.training_losses = []

    def __call__(self, ts_list, ys_list):
        baseline_list = []
        for j in range(self.num_sequences):
            start = j * self.num_rollouts
            end = start + self.num_rollouts

            sequence_baselines = self.predict_sequence(
                ts_list[start:end], ys_list[start:end]
            )

            baseline_list += sequence_baselines
        
        return baseline_list
    
    def predict_sequence(self, ts_list, ys_list):
        features_list = []
        for rollout_idx, (ts, ys) in enumerate(zip(ts_list, ys_list)):
            rollout_features = []
            for t in ts:
                features = [t]
                if self.feature_extractor is not None:
                    extra_features = self.feature_extractor(t, rollout_idx)
                    features.extend(extra_features)
                
                if len(features) > self.input_dim:
                    features = features[:self.input_dim]
                if len(features) < self.input_dim:
                    features.extend([0] * (self.input_dim - len(features)))

                rollout_features.append(features)
            features_list.append(np.array(rollout_features, dtype= np.float32))
            

        baseline_list = []

        for feature in features_list:
            with torch.no_grad():
                if self.state_mean is not None and self.state_std is not None:
                    #yeh define karna hai dekhna padega
                    feature = (feature - self.state_mean) / (self.state_std + 1e-8)
                
                print(f'shape of features going to the critic net in __call__ method are: [{feature.shape[0]}, {feature.shape[1]}]')
                
                feature_tensor = torch.FloatTensor(feature).to(self.device)

                value = self.network(feature_tensor)

                baseline_list.append(value)
        
        return baseline_list
    

    def update(self, states, returns, num_epochs=5, batch_size=64, alpha=0.05):

        if self.state_mean is None and self.state_std is None:
            self.state_mean = np.mean(states, axis=0)
            self.state_std = np.std(states, axis=0) + 1e-8

        else:
            self.state_mean = (1 - alpha) * self.state_mean + alpha * np.mean(states, axis=0)
            self.state_std = (1 - alpha) * self.state_std + alpha * np.std(states, axis=0) + 1e-8

        normalized_states = (states - self.state_mean) / self.state_std

        if self.returns_mean is None and self.returns_std is None:
            self.returns_mean = np.mean(returns, axis=0)
            self.returns_std = np.std(returns, axis=0) + 1e-8

        else:
            self.returns_mean = (1 - alpha) * self.returns_mean + alpha * np.mean(returns, axis=0)
            self.returns_std = (1 - alpha) * self.returns_std + alpha * np.std(returns, axis=0) + 1e-8

        normalized_returns = (returns - self.returns_mean) / self.returns_std

        state_tensors = torch.FloatTensor(normalized_states).to(self.device)
        return_tensors = torch.FloatTensor(normalized_returns).to(self.device)

        dataset = torch.utils.data.TensorDataset(state_tensors, return_tensors)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.network.train()
        losses = []

        for _ in range(num_epochs):
            for batch_states, batch_returns in dataloader:

                self.optimizer.zero_grad()

                predicted_vals = self.network(batch_states)

                loss = self.loss_fn(predicted_vals, batch_returns)

                loss.backward()

                self.optimizer.step()

                losses.append(loss.item())

        avg_loss = np.mean(losses)
        self.training_losses.append(avg_loss)

        return avg_loss
    
    def calc_loss(self, states, returns, alpha=0.05):
        if self.state_mean is None and self.state_std is None:
            self.state_mean = np.mean(states, axis=0)
            self.state_std = np.std(states, axis=0) + 1e-8

        else:
            self.state_mean = (1 - alpha) * self.state_mean + alpha * np.mean(states, axis=0)
            self.state_std = (1 - alpha) * self.state_std + alpha * np.std(states, axis=0) + 1e-8

        normalized_states = (states - self.state_mean) / self.state_std

        if self.returns_mean is None and self.returns_std is None:
            self.returns_mean = np.mean(returns, axis=0)
            self.returns_std = np.std(returns, axis=0) + 1e-8

        else:
            self.returns_mean = (1 - alpha) * self.returns_mean + alpha * np.mean(returns, axis=0)
            self.returns_std = (1 - alpha) * self.returns_std + alpha * np.std(returns, axis=0) + 1e-8

        normalized_returns = (returns - self.returns_mean) / self.returns_std

        print(f'return mean and std: {self.returns_mean, self.returns_std}')

        state_tensors = torch.FloatTensor(normalized_states).to(self.device)
        return_tensors = torch.FloatTensor(normalized_returns).to(self.device)

        predicted_vals = self.network(state_tensors)
        print(f'in calc_loss for combined the state tensors are of shape[{state_tensors.shape[0]}, {state_tensors.shape[1]}]')

        loss = self.loss_fn(predicted_vals, return_tensors)

        return loss


    
    def extract_features_from_observations(self, observations, times):
        features = []

        for obs, t in zip(observations, times):
            if hasattr(self, 'scheduler') and self.scheduler is not None:
                try:
                    dag_batch = utils.obs_to_pyg(obs)
                    dag_batch.to(self.scheduler.device)

                    with torch.no_grad():
                        h_dict = self.scheduler.encoder(dag_batch)
                        h_glob = h_dict['glob']

                    feature = h_glob.cpu().numpy().flatten()

                except Exception as e:
                    print(f'Error occured extracting h_glob: {e}')
                    feature = [t] + (self.input_dim - 1) * [0]
            
            else:
                print('the attribute scheduler not found or is None')
                feature.append(0)



            '''
            #This code is for manual selection of features from the observation

            feature =[]
            feature.append(t)
            print(f'got time {t}')

            #Use exec_mask more effectively
            if 'exec_mask' in obs:
                print('found exec mask')
                exec_mask = obs['exec_mask']
                num_active_executors = len(exec_mask)
                num_free_executors = np.sum(exec_mask) if isinstance(exec_mask, np.ndarray) else exec_mask.sum()
                feature.append(num_active_executors)
                feature.append(num_free_executors)
                print(f'num active execs: {num_active_executors}, num free exec: {num_free_executors}')
            else:
                feature.extend([0, 0])
                print('exec mask not found.')
            
            

            #Use stage_mask for better job information
            if 'stage_mask' in obs:
                stage_mask = obs['stage_mask']
                num_active_stages = np.sum(stage_mask) if isinstance(stage_mask, np.ndarray) else stage_mask.sum()
                feature.append(num_active_stages)
                print(f'stage mask found, num active stages: {num_active_stages}')
            else:
                feature.append(0)
                print('stage mask not found')

            # DAG features
            if 'dag_batch' in obs and hasattr(obs['dag_batch'], 'nodes'):
                print(f'nodes found in dag_batch')
                nodes = obs['dag_batch'].nodes
                if nodes.shape[0] > 0:
                    total_remaining_tasks = np.sum(nodes[:, 0])
                    avg_task_duration = np.mean(nodes[:, 1])
                    schedulable_stages = np.sum(nodes[:, 2].astype(bool))
                    
                    feature.extend([total_remaining_tasks, avg_task_duration, schedulable_stages])
                    print(f'total_remaining_tasks, avg_task_duration, schedulable_stages: {total_remaining_tasks, avg_task_duration, schedulable_stages}')
                else:
                    feature.extend([0, 0, 0])
                    print(f'dag_batch nodes not found appending 0')
            else:
                feature.extend([0, 0, 0])
                print('dag batch not found appending 0')

            '''
            

            # Pad or truncate to correct size
            if len(feature) > self.input_dim:
                feature = feature[:self.input_dim]
                print('more features thean dim')
            elif len(feature) < self.input_dim:
                feature.extend([0] * (self.input_dim - len(feature)))
                print('less features than dim')

            features.append(feature)

        return np.array(features, dtype=np.float32)


        
