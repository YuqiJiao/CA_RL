from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from numpy.random import RandomState
from sklearn.utils import check_random_state
from torch.utils.data import DataLoader, RandomSampler

from opendataval.dataloader.util import CatDataset
from opendataval.dataval.api import DataEvaluator
from cbp_code import ContinualBackprop


class CARL(DataEvaluator):


    def __init__(
            self,
            hidden_dim: int = 256,
            layer_number: int = 5,
            comb_dim: int = 128,
            rl_epochs: int = 2000,
            rl_batch_size: int = 512,
            lr: float = 0.025 ,
            threshold: float = 0.9,
            device: torch.device = torch.device("cpu"),
            random_state: RandomState = None,
    ):

        self.hidden_dim = hidden_dim
        self.layer_number = layer_number
        self.comb_dim = comb_dim
        self.device = device

        self.rl_epochs = rl_epochs
        self.rl_batch_size = rl_batch_size
        self.lr = lr
        self.threshold = threshold

        self.random_state = check_random_state(random_state)

    def input_data(
            self,
            x_train: torch.Tensor,
            y_train: torch.Tensor,
            x_valid: torch.Tensor,
            y_valid: torch.Tensor,
    ):

        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid

        self.num_points, [*self.feature_dim] = len(x_train), x_train[0].shape
        dim = [*self.feature_dim]
        [*self.label_dim] = (1,) if self.y_train.ndim == 1 else self.y_train[0].shape
        dim2 = [*self.label_dim]

        self.value_estimator = DataValueEstimatorRL(
            x_dim=np.prod(self.feature_dim),
            y_dim=np.prod(self.label_dim),
            hidden_dim=self.hidden_dim,
            layer_number=self.layer_number,
            comb_dim=self.comb_dim,
            random_state=self.random_state,
        ).to(self.device)

        return self

    def _evaluate_baseline_models(self, *args, **kwargs):

        # Final model
        self.final_model = self.pred_model.clone()

        # Train baseline model with input data
        self.ori_model = self.pred_model.clone()
        self.ori_model.fit(self.x_train, self.y_train, *args, **kwargs)

        # Trains validation model
        self.val_model = self.ori_model.clone()
        self.val_model.fit(self.x_valid, self.y_valid, *args, **kwargs)

        y_valid_hat = self.ori_model.predict(self.x_valid)
        self.valid_perf = self.evaluate(self.y_valid, y_valid_hat)
        if len(self.y_train[0]) == 2:
            self.hidden_dim = 100
            self.rl_batch_size = 32
            self.lr = 0.005

        y_pred = self.val_model.predict(self.x_train).cpu()

        self.y_pred_diff = torch.abs(self.y_train - y_pred)

    def train_data_values(self, *args, num_workers: int = 0, **kwargs):

        try:

            batch_size = min(self.rl_batch_size, len(self.x_train))
            self._evaluate_baseline_models(*args, **kwargs)

            # Solver
            cbp_learning = ContinualBackprop(net=self.value_estimator, loss=DveLoss(threshold=self.threshold),
                                             step_size=self.lr, opt='sgd', replacement_rate=0.0005, num_epochs=self.rl_epochs,
                                             maturity_threshold=200)


            gen = torch.Generator(self.device).manual_seed(self.random_state.tomaxint())
            cpu_gen = torch.Generator("cpu").manual_seed(self.random_state.tomaxint())

            data = CatDataset(self.x_train, self.y_train, self.y_pred_diff)
            rs = RandomSampler(data, False, self.rl_epochs * batch_size, generator=cpu_gen)  # True
            dataloader = DataLoader(
                data,
                batch_size,
                sampler=rs,
                generator=cpu_gen,
                pin_memory=True,
                num_workers=num_workers,
                persistent_workers=num_workers > 0,
            )
            if len(self.y_train[0]) == 2:
                base_perf = 0.5*self.valid_perf
            else:
                base_perf = self.valid_perf
            n_num = 0
            if len(self.y_train[0]) > 2:
                for x_batch, y_batch, y_hat_batch in tqdm.tqdm(dataloader):
                    n_num = n_num + 1
                    if n_num <= 1:  
                        nn = 0
                    else:
                        nn = 1 / 50
                    
                    x_batch_ve = x_batch.to(device=self.device)
                    y_batch_ve = y_batch.to(device=self.device)
                    y_hat_batch_ve = y_hat_batch.to(device=self.device)

                    pred_dataval = self.value_estimator(x_batch_ve, y_batch_ve, y_hat_batch_ve)
                    select_prob_s = []
                    dvrl_perf_s = []
                
                    while len(dvrl_perf_s) <= 8:
                
                        select_prob = torch.bernoulli(pred_dataval, generator=gen)
                        from collections import defaultdict
                        np.random.seed(42)

                        num_samples = y_batch_ve.shape[0]
                        class_to_indices = defaultdict(list)

                    
                        for idx, label in enumerate(y_batch_ve):
                            class_idx = torch.argmax(label).item()
                            class_to_indices[class_idx].append(idx)
  
                        from collections import defaultdict
                        select_prob_bool = select_prob.clone()

                        select_prob_bool = select_prob_bool.to(torch.bool)
                        select_prob_bool = select_prob_bool.squeeze(1)
                        y_act = y_batch_ve[select_prob_bool]
                        class_to_indices_act = defaultdict(list)

                        for idx, label in enumerate(y_act):
                            class_idx_act = torch.argmax(label).item()
                            class_to_indices_act[class_idx_act].append(idx)
                        expected_classes = set(class_to_indices.keys())  
                        actual_classes = set(class_to_indices_act.keys())  
                        if len(self.y_train[0]) == 200:
                            missing_classes1 = expected_classes - expected_classes
                        else:
                            missing_classes1 = expected_classes - actual_classes
                        
                        if len(missing_classes1) == 0:
                            
                            if select_prob.sum().item() == 0:  
                                pred_dataval = 0.5 * torch.ones_like(pred_dataval, requires_grad=True)
                                select_prob = torch.bernoulli(pred_dataval, generator=gen)
                            select_prob_s.append(select_prob)
                            new_model = self.pred_model.clone()
                            new_model.fit(
                                x_batch,
                                y_batch,
                                *args,
                                sample_weight=select_prob.detach().cpu(),  
                                **kwargs,
                            )

                            y_valid_hat = new_model.predict(self.x_valid)
                            dvrl_perf = self.evaluate(self.y_valid, y_valid_hat)
                            dvrl_perf_s.append(dvrl_perf)
                            del new_model
                    idx = np.argmin(dvrl_perf_s)

                    dvrl_perf = dvrl_perf_s[idx]
                    select_prob = select_prob_s[idx]
                    if len(dvrl_perf_s) >= 20000:
                        dvrl_perf_s.remove(dvrl_perf_s[idx])

                    dvrl_mean = sum(dvrl_perf_s) / len(dvrl_perf_s)
                    dvrl_mean = max(1.0 * base_perf, min(dvrl_mean, 1.5 * base_perf))
                    base_perf = (1 - nn) * base_perf + (nn) * dvrl_mean  
                    reward_curr = dvrl_perf - base_perf

                    cbp_learning.learn(x_batch_ve, y_batch_ve, y_hat_batch_ve, select_prob, reward_curr)
            
            weights = torch.zeros(0, 1, device=self.device)
            for x_batch, y_batch, y_hat_batch in DataLoader(
                    data, batch_size=self.rl_batch_size, shuffle=False
            ):
                
                x_batch = x_batch.to(device=self.device)
                y_batch = y_batch.to(device=self.device)
                y_hat_batch = y_hat_batch.to(device=self.device)

                data_values = self.value_estimator(x_batch, y_batch, y_hat_batch)
                weights = torch.cat([weights, data_values])

            self.final_model = self.pred_model.clone()
            self.final_model.fit(
                self.x_train,
                self.y_train,
                *args,
                sample_weight=weights.detach().cpu(), 
                **kwargs,
            )
        except Exception as e:
            print("Error occurred during execution:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {e}")
            print("Traceback:")
            import traceback
            traceback.print_exc()

        return self

    def evaluate_data_values(self) -> np.ndarray:

        y_valid_pred = self.final_model.predict(self.x_train).cpu()
        y_hat = torch.abs(self.y_train - y_valid_pred)
        response = torch.zeros(0, 1, device=self.device)

        with torch.no_grad():  

            data = CatDataset(self.x_train, self.y_train, y_hat)
            for x_batch, y_batch, y_hat_batch in DataLoader(
                    data, batch_size=self.rl_batch_size, shuffle=False
            ):
                x_batch = x_batch.to(device=self.device)
                y_batch = y_batch.to(device=self.device)
                y_hat_batch = y_hat_batch.to(device=self.device)

                data_values = self.value_estimator(x_batch, y_batch, y_hat_batch)
                response = torch.cat([response, data_values])

        return response.squeeze().numpy(force=True)

class DataValueEstimatorRL(nn.Module):


    def __init__(
            self,
            x_dim: int,
            y_dim: int,
            hidden_dim: int,
            layer_number: int,
            comb_dim: int,
            act_type: str = "relu",
            random_state: RandomState = None,
    ):
        super().__init__()

        if random_state is not None:
            torch.manual_seed(check_random_state(random_state).tomaxint())

        self.act_type = act_type
        act_layer = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "elu": nn.ELU,
            "selu": nn.SELU,
            "swish": nn.SiLU,
        }[act_type]

        self.backbone_layers = nn.ModuleList()
        in_dim = x_dim + y_dim

        self.backbone_layers.append(nn.Linear(in_dim, hidden_dim))
        self.backbone_layers.append(act_layer())

        for _ in range(layer_number - 3):
            self.backbone_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.backbone_layers.append(act_layer())

        self.backbone_layers.append(nn.Linear(hidden_dim, comb_dim))

        self.backbone_activation = act_layer()

        self.layers = self.backbone_layers  

        self.head = nn.Sequential(
            nn.Linear(comb_dim + y_dim, comb_dim),
            act_layer(),
            nn.Linear(comb_dim, 1),
            nn.Sigmoid(),  # selection probability
        )

    def predict(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            y_hat: torch.Tensor,
    ):
        """
        Forward pass with hidden activations exposed for CBP / GnT.
        Returns:
            output: selection probability
            features: list of hidden activations (backbone only)
        """

        x = x.flatten(start_dim=1)
        y = y.flatten(start_dim=1)
        y_hat = y_hat.flatten(start_dim=1)


        out = torch.cat((x, y), dim=1)
        features = []

        for layer in self.backbone_layers:
            out = layer(out)
            if isinstance(layer, nn.Linear):
                features.append(out)


        out = self.backbone_activation(out)


        out = torch.cat((out, y_hat), dim=1)
        out = self.head(out)

        return out, features

    def forward(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            y_hat: torch.Tensor,
    ):
        out, _ = self.predict(x, y, y_hat)
        return out


class DveLoss(nn.Module):


    def __init__(self, threshold: float = 0.9, exploration_weight: float = 1e3):
        super().__init__()
        self.threshold = threshold
        self.exploration_weight = exploration_weight

    def forward(
            self,
            pred_dataval: torch.Tensor,
            selector_input: torch.Tensor,
            reward_input: float,
    ) -> torch.Tensor:

        loss = F.binary_cross_entropy(pred_dataval, selector_input, reduction="sum")

        reward_loss = reward_input * loss
        search_loss = (  # Additional loss when VE is stuck outside threshold range
                F.relu(torch.mean(pred_dataval) - self.threshold)
                + F.relu((1 - self.threshold) - torch.mean(pred_dataval))
        )

        return reward_loss + (self.exploration_weight * search_loss)
