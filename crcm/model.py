from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .eval import cal_path_metrics, cal_mape, cal_r_squared, gen_prob_path
from .flow_sample import SAMPLING_RATE_EPS, alpha2gamma, gamma2demand, load_flow
from .net_load import PrismLoad, PrismRL
from .net_topo import load_net
from .reward import build_reward_model, compute_reward
from .traj_sample import TrajSample, load_train_test
from .utils import (
    experiment_dir,
    result_path,
    save_flow_csv,
    save_qd_csv,
    save_travel_time_csv,
    set_seed,
    write_json,
)


def grad_ll_reward(rho_real, rho_exp):
    return (rho_exp - rho_real).unsqueeze(-1)


def _require_finite_tensor(value, label, epoch, batch_id):
    if value is None or not bool(torch.isfinite(value).all()):
        raise FloatingPointError(
            f"{label} became non-finite at epoch {epoch}, batch {batch_id}. The optimizer step has not been applied."
        )


def _require_finite_gradients(named_parameters, label, epoch, batch_id):
    invalid = []
    for (name, parameter) in named_parameters:
        grad = parameter.grad
        if grad is not None and (not bool(torch.isfinite(grad).all())):
            invalid.append(name)
    if invalid:
        preview = ", ".join(invalid[:5])
        suffix = "" if len(invalid) <= 5 else f" (+{len(invalid) - 5} more)"
        raise FloatingPointError(
            f"{label} gradient became non-finite at epoch {epoch}, batch {batch_id}: {preview}{suffix}. The optimizer step has not been applied."
        )


def _require_finite_parameters(named_parameters, label, epoch, batch_id):
    invalid = [
        name
        for (name, parameter) in named_parameters
        if not bool(torch.isfinite(parameter).all())
    ]
    if invalid:
        preview = ", ".join(invalid[:5])
        suffix = "" if len(invalid) <= 5 else f" (+{len(invalid) - 5} more)"
        raise FloatingPointError(
            f"{label} parameter became non-finite after epoch {epoch}, batch {batch_id}: {preview}{suffix}."
        )


def _feature_paths(params):
    root = Path(params["data_root"]) / "net"
    return (root / "state_feature.pt", root / "action_feature.pt")


def _load_features(params):
    (state_path, action_path) = _feature_paths(params)
    state = torch.load(state_path, map_location=params["device"], weights_only=True).to(
        device=params["device"], dtype=params["tensor_dtype"]
    )
    action = torch.load(
        action_path, map_location=params["device"], weights_only=True
    ).to(device=params["device"], dtype=params["tensor_dtype"])
    return (state, action)


def init_model(params: Dict[str, Any], reward_params: Dict[str, Any]):
    set_seed(params["seed"], params.get("deterministic", True))
    net = load_net(params)
    (traj_all, traj_train, traj_test, traj_true) = load_train_test(params, net)
    net.add_demand_info(traj_all.gen_traj_od_df(net))
    (state_feature, action_feature) = _load_features(params)
    reward_params["state_in_dim"] = state_feature.size(1)
    reward_params["action_in_dim"] = action_feature.size(1)
    net.cal_edge_index()
    traj_all.gen_traj_info(net)
    net.cal_shortest_step()
    net.cal_step(traj_all, params)
    params["prism_horizon_definition"] = "D_s=(1+zeta)*max_i D(i,s)"
    params["prism_base_choice_stages"] = {
        str(d): int(value) for (d, value) in net.prism_base_stages.items()
    }
    params["prism_horizon_by_destination"] = {
        str(d): int(value) for (d, value) in net.T.items()
    }
    traj_train.gen_traj_info(net)
    traj_test.gen_traj_info(net)
    traj_true.gen_traj_info(net)
    flow_train = load_flow(params)
    flow_train.cal_beta(net, traj_train.traj_lids)
    flow_train.gen_traj_qd(net, traj_train)
    return (
        net,
        traj_all,
        traj_train,
        traj_test,
        traj_true,
        flow_train,
        state_feature,
        action_feature,
    )


def _make_loader(params, net, for_traj=False):
    return PrismRL(net) if for_traj else PrismLoad(net)


def _trajectory_terms(net, traj_batch, reward, params):
    loader = _make_loader(params, net, for_traj=True)
    loader.gen_traj_dest(traj_batch)
    loader.cal_traj_reward_mat(net, reward)
    loader.cal_traj_trans_prob(net)
    (psi_exp, rho_exp, ll) = loader.cal_exp_rho_ll(net, traj_batch)
    return (loader, psi_exp, rho_exp, ll)


def _flow_terms(net, flow_train, reward, gamma, params):
    loader = _make_loader(params, net, for_traj=False)
    loader.cal_reward_mat(net, reward)
    loader.cal_trans_prob(net)
    (lf_pred, lf_cov_pred) = loader.cal_lf_cov(net, flow_train, gamma)
    return (loader, lf_pred, lf_cov_pred)


def _observed_flow_slice(flow_train, lf_pred, lf_cov_pred, params, which="train"):
    lids = flow_train.obs_lids
    pred = lf_pred[lids]
    cov = lf_cov_pred[lids][:, lids]
    return (pred, cov, None)


def _network_metrics(flow_train, lf_pred, lf_cov_pred, net, params):
    true_flow = flow_train.mean_full_flow().to(
        device=lf_pred.device, dtype=lf_pred.dtype
    )
    true_flow_var = torch.clamp(
        torch.diagonal(flow_train.lf_cov_true).to(
            device=lf_pred.device, dtype=lf_pred.dtype
        ),
        min=0.0,
    )
    pred_flow_var = torch.clamp(torch.diagonal(lf_cov_pred), min=0.0)
    true_tt = flow_train.tt_true.to(device=lf_pred.device, dtype=lf_pred.dtype)
    true_tt_var = flow_train.tt_var_true.to(device=lf_pred.device, dtype=lf_pred.dtype)
    (pred_tt, pred_tt_var) = (true_tt, true_tt_var)
    (obs, unobs, full) = flow_train.flow_lid_groups(net.L)

    def one_group(lids):
        if not lids:
            return {
                "flow_mean_r2": None,
                "flow_mean_mape": None,
                "flow_variance_r2": None,
                "flow_variance_mape": None,
                "travel_time_mean_r2": None,
                "travel_time_mean_mape": None,
                "travel_time_variance_r2": None,
                "travel_time_variance_mape": None,
            }
        idx = torch.tensor(lids, device=lf_pred.device, dtype=torch.long)
        return {
            "flow_mean_r2": float(
                cal_r_squared(true_flow[idx], lf_pred[idx]).detach().cpu()
            ),
            "flow_mean_mape": float(
                cal_mape(true_flow[idx], lf_pred[idx]).detach().cpu()
            ),
            "flow_variance_r2": float(
                cal_r_squared(true_flow_var[idx], pred_flow_var[idx]).detach().cpu()
            ),
            "flow_variance_mape": float(
                cal_mape(true_flow_var[idx], pred_flow_var[idx]).detach().cpu()
            ),
            "travel_time_mean_r2": float(
                cal_r_squared(true_tt[idx], pred_tt[idx]).detach().cpu()
            ),
            "travel_time_mean_mape": float(
                cal_mape(true_tt[idx], pred_tt[idx]).detach().cpu()
            ),
            "travel_time_variance_r2": float(
                cal_r_squared(true_tt_var[idx], pred_tt_var[idx]).detach().cpu()
            ),
            "travel_time_variance_mape": float(
                cal_mape(true_tt_var[idx], pred_tt_var[idx]).detach().cpu()
            ),
        }

    return {
        "train_flow": one_group(obs),
        "test_flow": one_group(unobs),
        "full_flow": one_group(full),
    }


def _od_demand_metrics(flow_train, demand_pred, net):
    demand_true = flow_train.true_demand(net)
    if demand_true is None or demand_pred is None:
        return {"od_demand_r2": None, "od_demand_mape": None}
    demand_pred = demand_pred.to(device=demand_true.device, dtype=demand_true.dtype)
    return {
        "od_demand_r2": float(cal_r_squared(demand_true, demand_pred).detach().cpu()),
        "od_demand_mape": float(cal_mape(demand_true, demand_pred).detach().cpu()),
    }


def _train_network_metric(flow_train, lf_pred, lf_cov_pred, obs_lf_pred, net, params):
    true_flow = flow_train.mean_full_flow().to(
        device=lf_pred.device, dtype=lf_pred.dtype
    )
    idx = torch.tensor(flow_train.obs_lids, device=lf_pred.device, dtype=torch.long)
    pred_flow = lf_pred[idx]
    true_flow = true_flow[idx]
    true_tt = flow_train.tt_true.to(device=lf_pred.device, dtype=lf_pred.dtype)
    pred_tt = true_tt
    pred_tt = pred_tt[idx]
    true_tt = true_tt[idx]
    return {
        "flow_mean_r2": float(cal_r_squared(true_flow, pred_flow).detach().cpu()),
        "flow_mean_mape": float(cal_mape(true_flow, pred_flow).detach().cpu()),
        "travel_time_mean_r2": float(cal_r_squared(true_tt, pred_tt).detach().cpu())
        if true_tt is not None
        else None,
        "travel_time_mean_mape": float(cal_mape(true_tt, pred_tt).detach().cpu())
        if true_tt is not None
        else None,
    }


def _batch_generator(traj_train, batch_size, seed):
    indices = np.random.default_rng(seed).permutation(traj_train.len_trajs)
    for start in range(0, traj_train.len_trajs, batch_size):
        idx = indices[start : start + batch_size]
        yield TrajSample(
            [traj_train.traj_trans[i] for i in idx],
            [traj_train.traj_lids[i] for i in idx],
            [traj_train.traj_dests[i] for i in idx],
        )


def _global_batch_weight(batch_size: int, total_size: int) -> float:
    if total_size <= 0:
        raise ValueError("The training trajectory set must not be empty.")
    if batch_size <= 0 or batch_size > total_size:
        raise ValueError(
            f"Invalid trajectory batch size {batch_size} for total size {total_size}."
        )
    return batch_size / total_size


def _weighted_record_mean(records, key):
    valid = [(r[key], r["batch_size"]) for r in records if r.get(key) is not None]
    if not valid:
        return None
    total = sum((size for (_, size) in valid))
    return float(np.sum([value * size for (value, size) in valid]) / max(total, 1))


def _fmt_metric(value):
    return "n/a" if value is None else f"{value:.6g}"


def _init_gamma(flow_train, params):
    return nn.ParameterDict(
        {
            "od": nn.Parameter(
                alpha2gamma(flow_train.beta)
                * torch.ones(
                    flow_train.traj_q.numel(),
                    device=params["device"],
                    dtype=torch.float64,
                )
            )
        }
    )


def _theta_penalty(reward_model, params):
    terms = [
        torch.sum(p ** 2)
        for (name, p) in reward_model.named_parameters()
        if not name.endswith("bias")
    ]
    if not terms:
        first = next(reward_model.parameters())
        return torch.zeros((), device=first.device, dtype=first.dtype)
    return params["lambda_theta"] * torch.stack(terms).sum()


def _save_run_config(params, reward_params, reward_model):
    write_json(
        {
            "model_params": dict(params),
            "reward_params": dict(reward_params),
            "reward_parameter_count": int(
                sum((parameter.numel() for parameter in reward_model.parameters()))
            ),
        },
        result_path(params, "config.json"),
    )


def _early_stop_update(loss, epoch, params, state):
    previous = state.get("previous")
    relative_change = None
    if previous is not None:
        relative_change = abs(float(loss) - previous) / max(abs(previous), 1e-12)
    state["previous"] = float(loss)
    if not params.get("early_stop", False) or relative_change is None:
        state["stable"] = 0
        return (False, relative_change)
    if epoch + 1 >= int(
        params.get("early_stop_min_epoch", 20)
    ) and relative_change < float(params.get("early_stop_tol", 0.0001)):
        state["stable"] = state.get("stable", 0) + 1
    else:
        state["stable"] = 0
    stop = state["stable"] >= int(params.get("early_stop_patience", 3))
    return (stop, relative_change)


def train_crcm(
    net,
    traj_train,
    flow_train,
    state_feature,
    action_feature,
    params,
    reward_params,
    save=True,
):
    params["uses_sampling_rate_regularization"] = True
    params["od_initialization"] = "trajectory_sampling_rate_gamma"
    params["sampling_rate_floor"] = SAMPLING_RATE_EPS
    reward_model = build_reward_model(params, reward_params)
    if save:
        _save_run_config(params, reward_params, reward_model)
    gamma = _init_gamma(flow_train, params)
    opt_theta = torch.optim.Adam(reward_model.parameters(), lr=params["theta_lr"])
    opt_gamma = torch.optim.Adadelta(gamma.parameters(), lr=params["gamma_lr"])
    sched_theta = ReduceLROnPlateau(
        opt_theta,
        mode="min",
        factor=0.5,
        patience=6,
        min_lr=1e-05,
        threshold_mode="rel",
        threshold=0.005,
    )
    sched_gamma = ReduceLROnPlateau(
        opt_gamma,
        mode="min",
        factor=0.75,
        patience=6,
        min_lr=0.0001,
        threshold_mode="rel",
        threshold=0.005,
    )
    history = []
    stop_state = {}
    for epoch in range(params["num_epoch"]):
        reward_model.train()
        records = []
        batches = _batch_generator(
            traj_train, params["batch_size"], params["seed"] + epoch
        )
        for (batch_id, traj_batch) in enumerate(batches, start=1):
            traj_batch.gen_traj_info(net)
            opt_theta.zero_grad()
            opt_gamma.zero_grad()
            flow_r = compute_reward(
                reward_model, net, state_feature, action_feature, params["device"]
            )
            (_, psi_exp, rho_exp, ll) = _trajectory_terms(
                net, traj_batch, flow_r.detach().clone().to(torch.float64), params
            )
            n = max(traj_batch.len_trajs, 1)
            global_weight = _global_batch_weight(n, traj_train.len_trajs)
            traj_jac = grad_ll_reward(
                traj_batch.rho_real.to(device=flow_r.device, dtype=flow_r.dtype),
                rho_exp.to(device=flow_r.device, dtype=flow_r.dtype),
            )
            reward_leaf = flow_r.detach().clone().to(torch.float64).requires_grad_()
            (_, lf_pred, lf_cov_pred) = _flow_terms(
                net, flow_train, reward_leaf, gamma, params
            )
            (obs_pred, obs_cov, _) = _observed_flow_slice(
                flow_train, lf_pred, lf_cov_pred, params, "train"
            )
            l1 = -ll
            l2 = flow_train.cal_l2(obs_pred, obs_cov)
            l3_gamma_components = flow_train.cal_l3_components(gamma, params)
            l3_alpha = l3_gamma_components["alpha"]
            l3_od = l3_gamma_components["od"]
            l3_theta = _theta_penalty(reward_model, params)
            l3 = l3_alpha + l3_od + l3_theta
            weighted_l2 = global_weight * l2
            weighted_l3_alpha = global_weight * l3_alpha
            weighted_l3_od = global_weight * l3_od
            weighted_l3_theta = global_weight * l3_theta
            weighted_l3 = weighted_l3_alpha + weighted_l3_od + weighted_l3_theta
            weighted_global_loss = weighted_l2 + weighted_l3
            weighted_global_loss.backward()
            _require_finite_tensor(
                reward_leaf.grad, "Flow-loss reward", epoch, batch_id
            )
            _require_finite_gradients(
                gamma.named_parameters(), "Sampling-rate", epoch, batch_id
            )
            flow_grad = reward_leaf.grad[:-1].to(
                device=flow_r.device, dtype=flow_r.dtype
            )
            flow_r[:-1].backward(traj_jac.to(flow_grad) + flow_grad)
            _require_finite_gradients(
                reward_model.named_parameters(), "Reward-model", epoch, batch_id
            )
            _require_finite_gradients(
                gamma.named_parameters(), "Sampling-rate", epoch, batch_id
            )
            demand_for_metric = gamma2demand(net, flow_train, gamma)
            opt_theta.step()
            opt_gamma.step()
            _require_finite_parameters(
                reward_model.named_parameters(), "Reward-model", epoch, batch_id
            )
            _require_finite_parameters(
                gamma.named_parameters(), "Sampling-rate", epoch, batch_id
            )
            loss = l1 + weighted_global_loss
            network_metric = _train_network_metric(
                flow_train,
                lf_pred.detach(),
                lf_cov_pred.detach(),
                obs_pred.detach(),
                net,
                params,
            )
            od_metric = _od_demand_metrics(flow_train, demand_for_metric, net)
            rec = {
                "epoch": epoch,
                "batch": batch_id,
                "batch_size": n,
                "global_weight": global_weight,
                "loss": float(loss.detach().cpu()),
                "l1": float(l1.detach().cpu()),
                "l2": float(weighted_l2.detach().cpu()),
                "l3": float(weighted_l3.detach().cpu()),
                "l3_alpha": float(weighted_l3_alpha.detach().cpu()),
                "l3_od": float(weighted_l3_od.detach().cpu()),
                "l3_theta": float(weighted_l3_theta.detach().cpu()),
                "train_traj_anll": float((l1 / n).detach().cpu()),
                "train_traj_r2": float(
                    cal_r_squared(traj_batch.psi_real, psi_exp).detach().cpu()
                ),
                "train_flow_mean_r2": network_metric["flow_mean_r2"],
                "train_flow_mean_mape": network_metric["flow_mean_mape"],
                "train_travel_time_mean_r2": network_metric["travel_time_mean_r2"],
                "train_travel_time_mean_mape": network_metric["travel_time_mean_mape"],
                "train_od_demand_r2": od_metric["od_demand_r2"],
                "train_od_demand_mape": od_metric["od_demand_mape"],
            }
            records.append(rec)
            print(
                f"Epoch {epoch}, batch {batch_id}, batch loss {rec['loss']:.6g}, global_weight {rec['global_weight']:.6g}, l1 {rec['l1']:.6g}, l2 {rec['l2']:.6g}, l3 {rec['l3']:.6g}, train_traj_anll {rec['train_traj_anll']:.6g}, train_traj_r2 {rec['train_traj_r2']:.6g}, train_flow_mean_r2 {rec['train_flow_mean_r2']:.6g}, train_flow_mean_mape {rec['train_flow_mean_mape']:.6g}, train_travel_time_mean_r2 {_fmt_metric(rec['train_travel_time_mean_r2'])}, train_travel_time_mean_mape {_fmt_metric(rec['train_travel_time_mean_mape'])}. train_od_demand_r2 {_fmt_metric(rec['train_od_demand_r2'])}, train_od_demand_mape {_fmt_metric(rec['train_od_demand_mape'])}."
            )
        total_n = max(sum((r["batch_size"] for r in records)), 1)
        global_weight_sum = float(np.sum([r["global_weight"] for r in records]))
        if not np.isclose(global_weight_sum, 1.0, rtol=0.0, atol=1e-12):
            raise RuntimeError(
                f"Global loss was not allocated exactly once in epoch {epoch}: weight sum={global_weight_sum}."
            )
        epoch_metrics = {
            "epoch": epoch,
            "loss": float(np.sum([r["loss"] for r in records])),
            "l1": float(np.sum([r["l1"] for r in records])),
            "l2": float(np.sum([r["l2"] for r in records])),
            "l3": float(np.sum([r["l3"] for r in records])),
            "l3_alpha": float(np.sum([r["l3_alpha"] for r in records])),
            "l3_od": float(np.sum([r["l3_od"] for r in records])),
            "l3_theta": float(np.sum([r["l3_theta"] for r in records])),
            "global_weight_sum": global_weight_sum,
            "train_traj_anll": float(
                np.sum([r["train_traj_anll"] * r["batch_size"] for r in records])
                / total_n
            ),
            "train_traj_r2": float(
                np.sum([r["train_traj_r2"] * r["batch_size"] for r in records])
                / total_n
            ),
            "train_flow_mean_r2": _weighted_record_mean(records, "train_flow_mean_r2"),
            "train_flow_mean_mape": _weighted_record_mean(
                records, "train_flow_mean_mape"
            ),
            "train_travel_time_mean_r2": _weighted_record_mean(
                records, "train_travel_time_mean_r2"
            ),
            "train_travel_time_mean_mape": _weighted_record_mean(
                records, "train_travel_time_mean_mape"
            ),
            "train_od_demand_r2": _weighted_record_mean(records, "train_od_demand_r2"),
            "train_od_demand_mape": _weighted_record_mean(
                records, "train_od_demand_mape"
            ),
        }
        sched_theta.step(epoch_metrics["loss"])
        sched_gamma.step(epoch_metrics["loss"])
        epoch_metrics["theta_lr"] = float(opt_theta.param_groups[0]["lr"])
        epoch_metrics["gamma_lr"] = float(opt_gamma.param_groups[0]["lr"])
        print("Epoch summary:", epoch_metrics)
        epoch_metrics["batches"] = records
        history.append(epoch_metrics)
        (stop, rel_change) = _early_stop_update(
            epoch_metrics["loss"], epoch, params, stop_state
        )
        epoch_metrics["relative_loss_change"] = rel_change
        if stop:
            epoch_metrics["early_stopped"] = True
            print(
                f"Early stopping at epoch {epoch}: relative loss change={rel_change:.6g}."
            )
            break
    if save:
        out = experiment_dir(params)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(reward_model.state_dict(), result_path(params, "reward.pth"))
        write_json({"history": history}, result_path(params, "iteration.json"))
        reward_model.eval()
        with torch.no_grad():
            reward = compute_reward(
                reward_model, net, state_feature, action_feature, params["device"]
            )
            (_, lf_pred, lf_cov_pred) = _flow_terms(
                net, flow_train, reward.to(torch.float64), gamma, params
            )
        demand = gamma2demand(net, flow_train, gamma)
        flow_variance = torch.clamp(torch.diagonal(lf_cov_pred), min=0.0)
        tt_mean = flow_train.tt_true.to(device=lf_pred.device, dtype=lf_pred.dtype)
        tt_variance = flow_train.tt_var_true.to(
            device=lf_pred.device, dtype=lf_pred.dtype
        )
        save_travel_time_csv(
            tt_mean, tt_variance, result_path(params, "travel_time.csv")
        )
        save_flow_csv(
            lf_pred, result_path(params, "flow.csv"), flow_variance=flow_variance
        )
        save_qd_csv(net, demand, result_path(params, "qd.csv"))
    return (reward_model, gamma, history)


def _eval_traj_metrics(net, traj, reward, params):
    traj.gen_traj_info(net)
    loader = _make_loader(params, net, for_traj=True)
    loader.gen_traj_dest(traj)
    loader.cal_traj_reward_mat(net, reward)
    loader.cal_traj_trans_prob(net)
    (psi_exp, _, ll) = loader.cal_exp_rho_ll(net, traj)
    od2traj_id = traj.gen_od2traj_id()
    paths = gen_prob_path(
        net, loader, od2traj_id, -1, rng=np.random.default_rng(int(params["seed"]))
    )
    path_metrics = cal_path_metrics(net, traj, od2traj_id, paths)
    return {
        "anll": float((-ll / max(traj.len_trajs, 1)).detach().cpu()),
        "traj_r2": float(cal_r_squared(traj.psi_real, psi_exp).detach().cpu()),
        **path_metrics,
    }


def test_model(
    net,
    traj_train,
    traj_test,
    traj_true,
    flow_train,
    state_feature,
    action_feature,
    params,
    reward_params,
    reward_model=None,
    gamma=None,
    save=True,
):
    set_seed(params["seed"], params.get("deterministic", True))
    if reward_model is None:
        reward_model = build_reward_model(params, reward_params)
        reward_model.load_state_dict(
            torch.load(
                result_path(params, "reward.pth"),
                map_location=params["device"],
                weights_only=True,
            )
        )
    reward_model.eval()
    with torch.no_grad():
        reward = compute_reward(
            reward_model, net, state_feature, action_feature, params["device"]
        ).to(torch.float64)
        lf_pred = lf_cov_pred = None
    res = {
        "train_traj": _eval_traj_metrics(net, traj_train, reward, params),
        "test_traj": _eval_traj_metrics(net, traj_test, reward, params),
        "true_traj": _eval_traj_metrics(net, traj_true, reward, params),
    }
    (_, lf_pred, lf_cov_pred) = _flow_terms(net, flow_train, reward, gamma, params)
    res.update(_network_metrics(flow_train, lf_pred, lf_cov_pred, net, params))
    demand_pred = gamma2demand(net, flow_train, gamma)
    res["od_demand"] = _od_demand_metrics(flow_train, demand_pred, net)
    if save:
        if lf_pred is not None and lf_cov_pred is not None:
            flow_variance = torch.clamp(torch.diagonal(lf_cov_pred), min=0.0)
            tt_mean = flow_train.tt_true.to(device=lf_pred.device, dtype=lf_pred.dtype)
            tt_variance = flow_train.tt_var_true.to(
                device=lf_pred.device, dtype=lf_pred.dtype
            )
            save_flow_csv(
                lf_pred, result_path(params, "flow.csv"), flow_variance=flow_variance
            )
            save_travel_time_csv(
                tt_mean, tt_variance, result_path(params, "travel_time.csv")
            )
        write_json(res, result_path(params, "res.json"))
    return res
