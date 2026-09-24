from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATv2Conv
except Exception as exc:
    GATv2Conv = None


class FeatureMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim, track_running_stats=False)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout)
        self._init_weights()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        return self.fc2(x)

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)


class DNNReward(nn.Module):
    def __init__(self, reward_params):
        super().__init__()
        self.state_mlp = FeatureMLP(
            reward_params["state_in_dim"],
            reward_params["state_hidden_dim"],
            reward_params["state_out_dim"],
            reward_params["dropout"],
        )
        self.action_mlp = FeatureMLP(
            reward_params["action_in_dim"],
            reward_params["action_hidden_dim"],
            reward_params["action_out_dim"],
            reward_params["dropout"],
        )
        self.trans_mlp = FeatureMLP(
            reward_params["state_out_dim"] + reward_params["action_out_dim"],
            reward_params["trans_hidden_dim"],
            1,
            reward_params["dropout"],
        )

    def raw_utility(self, net, state_feature, action_feature):
        state_feature_out = self.state_mlp(state_feature)
        action_feature_out = self.action_mlp(action_feature)
        trans_feature_in = torch.cat([state_feature_out, action_feature_out], dim=1)
        u = self.trans_mlp(trans_feature_in)
        self.u = u
        return u

    def forward(self, net, state_feature, action_feature):
        u = self.raw_utility(net, state_feature, action_feature)
        return -F.softplus(u)


class FeatureGAT(nn.Module):
    def __init__(
        self, state_dim, action_dim, hidden_dim, output_dim, heads=4, dropout=0.2
    ):
        super().__init__()
        if GATv2Conv is None:
            raise ImportError("torch_geometric is required for the reward model.")
        self.gat1 = GATv2Conv(
            in_channels=state_dim,
            out_channels=hidden_dim,
            heads=heads,
            edge_dim=action_dim,
            dropout=dropout,
            concat=True,
        )
        self.gat2 = GATv2Conv(
            in_channels=hidden_dim * heads,
            out_channels=output_dim,
            heads=heads,
            edge_dim=action_dim,
            dropout=dropout,
            concat=False,
        )

    def forward(self, x, edge_index, edge_attr):
        x = self.gat1(x, edge_index, edge_attr)
        x = self.gat2(x, edge_index, edge_attr)
        return x


class GATReward(nn.Module):
    def __init__(self, reward_params):
        super().__init__()
        self.gat = FeatureGAT(
            reward_params["state_in_dim"],
            reward_params["action_in_dim"],
            reward_params["state_hidden_dim"],
            reward_params["state_out_dim"],
            reward_params["heads"],
            reward_params["dropout"],
        )
        self.trans_mlp = FeatureMLP(
            reward_params["state_out_dim"] + reward_params["action_in_dim"],
            reward_params["trans_hidden_dim"],
            1,
            reward_params["dropout"],
        )

    def raw_utility(self, state_feature, edge_index, edge_attr=None):
        state_feature_out = self.gat(state_feature, edge_index, edge_attr)
        trans_feature_in = torch.cat(
            [state_feature_out[edge_index[1], :], edge_attr], dim=1
        )
        g = self.trans_mlp(trans_feature_in)
        self.g = g
        return g

    def forward(self, state_feature, edge_index, edge_attr=None):
        g = self.raw_utility(state_feature, edge_index, edge_attr)
        return -F.softplus(g)


class DNNGATReward(nn.Module):
    def __init__(self, dnn_model, gat_model):
        super().__init__()
        self.dnn = dnn_model
        self.gat = gat_model

    def forward(self, net, state_feature, action_feature):
        u = self.dnn.raw_utility(net, state_feature[net.head_lids, :], action_feature)
        g = self.gat.raw_utility(state_feature, net.edge_index, action_feature)
        self.u = u
        self.g = g
        return -F.softplus(u + g)


def build_reward_model(model_params, reward_params):
    model = DNNGATReward(DNNReward(reward_params), GATReward(reward_params))
    return model.to(device=model_params["device"], dtype=model_params["tensor_dtype"])


def compute_reward(reward_model, net, state_feature, action_feature, device):
    r = reward_model(net, state_feature, action_feature)
    return torch.cat([r, torch.zeros(1, 1, device=device, dtype=r.dtype)], dim=0)
