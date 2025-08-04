import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from schedulers.decima import utils
from collections import OrderedDict


class MetaCriticNetwork(nn.Module):
    def __init__(self, input_dim: int = 16, hidden_dim1: int = 64, hidden_dim2: int = 32):
        super(MetaCriticNetwork, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, 1)
        )

    def forward(self, features: Tensor, params: OrderedDict = None) -> Tensor:
        if params is None:
            return self.model(features).squeeze(-1)
        else:
            x = features
            params_items = list(params.items())
            num_layers = len(self.model) // 2

            for i in range(num_layers):
                weight_key, weight = params_items[i * 2]
                bias_key, bias = params_items[i*2 + 1]
                x = nn.functional.linear(x, weight, bias)
                x = nn.functional.relu(x)

            # final layer:
            weight_key, weight = params_items[-2]
            bias_key, bias = params_items[-1]
            x = nn.functional.linear(x, weight, bias)

            return x

    
class MetaCritic:
    def __init__(self, input_dim: int = 16, hidden_dim1: int = 64, hidden_dim2: int = 32, 
                 meta_lr: float = 3e-4, inner_lr: float = 3e-4, num_sequences: int = 4, num_rollouts: int = 4,
                 feature_extractor = None):
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.lr = meta_lr
        self.inner_lr = inner_lr
        self.num_sequences = num_sequences
        self.num_rollouts = num_rollouts
        self.feature_extractor = feature_extractor

        self.network = MetaCriticNetwork(self.input_dim, self.hidden_dim1, self.hidden_dim2)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()
        #update for max grad norm.

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network.to(self.device)

        self.state_mean = None
        self.state_std = None
        self.returns_mean = None
        self.returns_std = None
        self.training_losses = []

    def __call__(self, ts_list, obs_list):
        assert len(ts_list) == len(obs_list), f"length mismatch between times: {len(ts_list)} and obs: {len(obs_list)}"
        baselines = self.predict_baseline(ts_list, obs_list)
        return baselines

    def predict_baseline(self, ts_list, obs_list):

        baselines = []

        features = self.extract_features_from_obs(obs_list, ts_list)

        with torch.no_grad():
            if self.state_mean is not None and self.state_std is not None:
                features = (features - self.state_mean) / (self.state_std + 1e-8)
            
            features_tensor = torch.FloatTensor(features)

            values = self.network(features_tensor)
            values = values.squeeze(-1)

        baselines = values.squeeze(-1).cpu().numpy()

        return baselines
    
    def compute_loss_and_grad(self, features, returns, params: OrderedDict):
        
        predicted_values = self.network(features, params=params)

        if isinstance(returns, torch.Tensor):
            if returns.dim() == 1:
                returns = returns.unsqueeze(-1)
        else:
            returns = torch.tensor(returns, dtype=torch.float32)
            if returns.dim() == 1:
                returns = returns.unsqueeze(-1)

        loss = self.loss_fn(predicted_values, returns)
        grad = torch.autograd.grad(loss, params.values(), create_graph=True)

        return loss, grad
    
    def _param_update(self, params: OrderedDict, grads: tuple, lr: float):
        adapted_params = OrderedDict()

        param_keys = list(params.keys())

        if len(param_keys) != len(grads):
            raise ValueError('number of params and grads dont match')
        
        for key, grad in zip(param_keys, grads):
            adapted_params[key] = params[key] - (lr * grad)

        return adapted_params
    
    def _adapt(self, features, returns):

        original_params = OrderedDict(self.network.named_parameters())

        loss, grads = self.compute_loss_and_grad(features, returns, original_params)

        adapted_params = self._param_update(original_params, grads, self.inner_lr)

        return adapted_params
    
    def update(self, time_a, time_b, obs_a, obs_b, value_a, value_b, alpha:float = 0.05, batch_size: int = 64):

        features_A, features_B, return_A, return_B = self._normalized_tensors(time_a, time_b, obs_a, obs_b, value_a, value_b, alpha)

        dataset = torch.utils.data.TensorDataset(features_A, features_B, return_A, return_B)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.network.train()
        losses = []
        ## Adapt A and then predict B
        for feature_a, feature_b, returns_a, returns_b in dataloader:
            params_adapted_on_a = self._adapt(feature_a, returns_a)
            predicted_values_b = self.network(feature_b, params=params_adapted_on_a)
            loss_a = self.loss_fn(predicted_values_b, returns_b)

            ## Adapt B and then predict A
            params_adapted_on_b = self._adapt(feature_b, returns_b)
            predicted_values_a = self.network(feature_a, params=params_adapted_on_b)
            loss_b = self.loss_fn(predicted_values_a, returns_a)

            self.optimizer.zero_grad()
            combined_loss = loss_a + loss_b
            combined_loss.backward()
            self.optimizer.step()

            losses.append(combined_loss.item())

        avg_loss = np.mean(losses)
        self.training_losses.append(avg_loss)

        return avg_loss

    def _normalized_tensors(self, time_a, time_b, obs_a, obs_b, value_a, value_b, alpha=0.05):

        features_a = self.extract_features_from_obs(obs_a, time_a)
        features_b = self.extract_features_from_obs(obs_b, time_b)
        features = np.append(features_a, features_b)
        returns = np.append(value_a, value_b)

        if self.state_mean is None and self.state_std is None:
            self.state_mean = np.mean(features)
            self.state_std = np.std(features) + 1e-8
        else:
            self.state_mean = ((1 - alpha) * self.state_mean) + (alpha * np.mean(features))
            self.state_std = ((1 - alpha) * self.state_std) + (alpha * np.std(features))

        if self.returns_mean is None and self.returns_std is None:
            self.returns_mean = np.mean(returns)
            self.returns_std = np.std(returns) + 1e-8
        else:
            self.returns_mean = ((1 - alpha) * self.returns_mean) + (alpha * np.mean(returns))
            self.returns_std = ((1 - alpha) * self.returns_std) + (alpha * np.std(returns))

        normalized_features_a = (features_a - self.state_mean) / (self.state_std + 1e-8)
        normalized_features_b = (features_b - self.state_mean) / (self.state_std + 1e-8)
        normalized_returns_a = (value_a - self.returns_mean) / (self.returns_std + 1e-8)
        normalized_returns_b = (value_b - self.returns_mean) / (self.returns_std + 1e-8)

        feat_a = torch.FloatTensor(normalized_features_a)
        feat_b = torch.FloatTensor(normalized_features_b)
        ret_a = torch.FloatTensor(normalized_returns_a)
        ret_b = torch.FloatTensor(normalized_returns_b)

        if ret_a.dim() == 1:
            ret_a = ret_a.unsqueeze(-1)  # [batch_size] -> [batch_size, 1]
        if ret_b.dim() == 1:
            ret_b = ret_b.unsqueeze(-1)

        return feat_a, feat_b, ret_a, ret_b
    
    def extract_features_from_obs(self, observations, times):
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

            # Pad or truncate to correct size
            if len(feature) > self.input_dim:
                feature = feature[:self.input_dim]
                print('more features thean dim')
            elif len(feature) < self.input_dim:
                feature.extend([0] * (self.input_dim - len(feature)))
                print('less features than dim')

            features.append(feature)

        return np.array(features, dtype=np.float32)








        




       