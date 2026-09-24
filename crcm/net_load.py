from __future__ import annotations
import math
import torch
from .flow_sample import gamma2alpha


def _require_finite(label, *tensors):
    for (index, tensor) in enumerate(tensors):
        if bool(torch.isfinite(tensor).all()):
            continue
        finite = tensor[torch.isfinite(tensor)]
        finite_range = (
            "no finite values"
            if finite.numel() == 0
            else f"finite_min={float(finite.min().detach().cpu()):.6g}, finite_max={float(finite.max().detach().cpu()):.6g}"
        )
        raise FloatingPointError(
            f"{label} tensor {index} became non-finite ({finite_range})."
        )


class PrismLoad:
    def __init__(self, net):
        self.state_space = {d: {t: [] for t in range(net.T[d] + 1)} for d in net.dests}
        for (origin, dest) in net.od2did:
            self.state_space[dest][0].append(net.origin_virtual_lid[origin])
        for (d, steps) in net.dest_steps.items():
            for (lid, Dt) in steps.items():
                max_t = net.T[d] - Dt
                for t in range(1, max_t + 1):
                    self.state_space[d][t].append(lid)
        for d in net.dests:
            self.state_space[d][0] = sorted(set(self.state_space[d][0]))
        if not hasattr(net, "_prism_topo_cache"):
            net._prism_topo_cache = {}
        if not hasattr(net, "_prism_topo_tensor_cache"):
            net._prism_topo_tensor_cache = {}
        self._topo_cache = net._prism_topo_cache
        self._topo_tensor_cache = net._prism_topo_tensor_cache

    def _stage_topology(self, net, d, t):
        key = (d, t)
        if key in self._topo_cache:
            return self._topo_cache[key]
        (row_ids, col_ids, value_ids) = ([], [], [])
        states_next = set(self.state_space[d][t + 1])
        for from_lid in self.state_space[d][t]:
            if t > 0 and from_lid in net.nid2upstrm.get(d, []):
                row_ids.append(from_lid)
                col_ids.append(net.terminal_lid)
                value_ids.append(-1)
                continue
            valid_to_lids = [
                to_lid
                for to_lid in net.lid2dstrm.get(from_lid, [])
                if to_lid in states_next
            ]
            if valid_to_lids:
                row_ids.extend([from_lid] * len(valid_to_lids))
                col_ids.extend(valid_to_lids)
                value_ids.extend(
                    [net.ft2tid[from_lid, to_lid] for to_lid in valid_to_lids]
                )
        self._topo_cache[key] = (row_ids, col_ids, value_ids)
        return self._topo_cache[key]

    def _stage_indices(self, net, d, t, device):
        device = torch.device(device)
        key = (int(d), int(t), device.type, device.index)
        if key not in self._topo_tensor_cache:
            (row_ids, col_ids, value_ids) = self._stage_topology(net, d, t)
            idx = torch.tensor([row_ids, col_ids], device=device, dtype=torch.long)
            value_idx = torch.tensor(value_ids, device=device, dtype=torch.long)
            self._topo_tensor_cache[key] = (idx, value_idx)
        return self._topo_tensor_cache[key]

    def _build_stage_matrices(self, net, reward, dests):
        self.torch_params = {"dtype": reward.dtype, "device": reward.device}
        self.reward = reward
        self.reward_dests = tuple(dests)

    def cal_reward_mat(self, net, reward):
        self._build_stage_matrices(net, reward, net.dests)

    def _cal_trans_prob_for_dests(self, net, dests):
        S = net.num_choice_states
        size = (S, S)
        log_floor = math.log(torch.finfo(self.torch_params["dtype"]).tiny)
        self.log_z_mat = {d: {} for d in dests}
        self.p_mat = {d: {} for d in dests}
        for d in dests:
            log_z_next = torch.full((S,), log_floor, **self.torch_params)
            log_z_next[net.terminal_lid] = 0.0
            self.log_z_mat[d][net.T[d]] = log_z_next
            for t in range(net.T[d] - 1, -1, -1):
                (row_ids, col_ids, value_ids) = self._stage_topology(net, d, t)
                if not row_ids:
                    raise RuntimeError(
                        f"Destination {d} has no feasible prism transition at stage {t}."
                    )
                (idx, value_idx) = self._stage_indices(net, d, t, self.reward.device)
                edge_scores = self.reward[value_idx, 0] + log_z_next[idx[1]]
                row_max = torch.full(
                    (S,), -torch.inf, **self.torch_params
                ).scatter_reduce(
                    0, idx[0], edge_scores, reduce="amax", include_self=True
                )
                exp_sum = torch.zeros(S, **self.torch_params).scatter_add(
                    0, idx[0], torch.exp(edge_scores - row_max[idx[0]])
                )
                valid_row = exp_sum > 0
                log_z = torch.where(
                    valid_row,
                    row_max
                    + torch.log(
                        torch.clamp(exp_sum, min=torch.finfo(exp_sum.dtype).tiny)
                    ),
                    torch.full_like(row_max, log_floor),
                )
                log_z = log_z.clone()
                log_z[net.terminal_lid] = 0.0
                p_values = torch.exp(edge_scores - log_z[idx[0]])
                _require_finite(
                    f"Recursive-logit probabilities for destination {d}, stage {t}",
                    p_values,
                )
                self.log_z_mat[d][t] = log_z
                self.p_mat[d][t] = torch.sparse_coo_tensor(
                    idx, p_values, size=size, device=log_z.device, dtype=log_z.dtype
                ).coalesce()
                log_z_next = log_z

    def cal_trans_prob(self, net):
        self._cal_trans_prob_for_dests(net, net.dests)

    def _state_demand(self, net, flow_sample, gamma):
        g = gamma["od"]
        demand = flow_sample.traj_q.to(device=g.device, dtype=g.dtype) / gamma2alpha(g)
        return net.od_vector_to_state_demand(demand)

    def _moments_for_dests(self, net, qd_getter, dests):
        S = net.num_choice_states
        flow_sum = torch.zeros(S, **self.torch_params)
        cov_mat = torch.zeros((S, S), **self.torch_params)
        for d in dests:
            horizon = net.T[d]
            p_dense = [self.p_mat[d][t].to_dense() for t in range(horizon)]
            S_mat = [None] * horizon
            S_mat[horizon - 1] = p_dense[horizon - 1]
            for t in range(net.T[d] - 2, -1, -1):
                S_mat[t] = p_dense[t] + torch.sparse.mm(self.p_mat[d][t], S_mat[t + 1])
            flow_t = qd_getter(d).to(**self.torch_params)
            flow_sum = flow_sum + flow_t
            for t in range(net.T[d]):
                F = flow_t.unsqueeze(1) * S_mat[t]
                cov_mat = cov_mat + F + F.T
                flow_t = flow_t @ p_dense[t]
                flow_sum = flow_sum + flow_t
        L = net.num_real_links
        return (flow_sum[:L], (cov_mat + torch.diag(flow_sum))[:L, :L])

    def cal_lf_cov(self, net, flow_sample, gamma):
        qd = self._state_demand(net, flow_sample, gamma)
        return self._moments_for_dests(net, lambda d: qd[d], net.dests)


class PrismRL(PrismLoad):
    def gen_traj_dest(self, traj):
        self.traj_dests = sorted(set(traj.traj_dests))

    def cal_traj_reward_mat(self, net, reward):
        self._build_stage_matrices(net, reward, self.traj_dests)

    def cal_traj_trans_prob(self, net):
        self._cal_trans_prob_for_dests(net, self.traj_dests)

    def cal_exp_rho_ll(self, net, traj):
        flow_sum = torch.zeros(net.num_choice_states, **self.torch_params)
        rho_sum = torch.zeros(len(net.tids), **self.torch_params)
        ll = 0
        (from_lids, to_lids) = (net.tail_lids, net.head_lids)
        for d in self.traj_dests:
            flow_t = traj.qd[d].to(device=flow_sum.device, dtype=flow_sum.dtype)
            flow_sum = flow_sum + flow_t
            for t in range(net.T[d]):
                p_dense = self.p_mat[d][t].to_dense()
                rho_sum = rho_sum + flow_t[from_lids] * p_dense[from_lids, to_lids]
                flow_t = flow_t @ p_dense
                flow_sum = flow_sum + flow_t
                if t in traj.traj_steps.get(d, {}):
                    (a, b) = traj.traj_steps[d][t]
                    ll = ll + torch.sum(
                        torch.log(torch.clamp(p_dense[a, b], min=1e-30))
                    )
        return (flow_sum[: net.num_real_links], rho_sum, ll)
