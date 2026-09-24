from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import torch
from .utils import to_device

SAMPLING_RATE_EPS = 0.0001


def gamma2alpha(gamma):
    width = 1.0 - 2.0 * SAMPLING_RATE_EPS
    return SAMPLING_RATE_EPS + width * torch.sigmoid(gamma)


def alpha2gamma(alpha):
    width = 1.0 - 2.0 * SAMPLING_RATE_EPS
    dtype_eps = torch.finfo(alpha.dtype).eps
    scaled = (alpha - SAMPLING_RATE_EPS) / width
    scaled = torch.clamp(scaled, min=dtype_eps, max=1.0 - dtype_eps)
    return torch.logit(scaled)


def gamma2demand(net, flow_sample, gamma: torch.nn.ParameterDict):
    g = gamma["od"]
    traj_q = flow_sample.traj_q.to(device=g.device, dtype=g.dtype)
    return traj_q / gamma2alpha(g)


class FlowSample:
    def __init__(
        self,
        obs_lf_sample,
        lf_true,
        obs_lids,
        params: dict,
        obs_lf_true=None,
        lf_cov_true=None,
        tt_true=None,
        tt_var_true=None,
        q_prior=None,
        demand_df=None,
    ):
        self.torch_params = {
            "device": params["device"],
            "dtype": params["tensor_dtype"],
        }
        self.obs_lf_sample = obs_lf_sample.to(**self.torch_params)
        self.lf_true = lf_true.to(**self.torch_params) if lf_true is not None else None
        self.lf_cov_true = (
            lf_cov_true.to(**self.torch_params) if lf_cov_true is not None else None
        )
        self.tt_true = tt_true.to(**self.torch_params) if tt_true is not None else None
        self.tt_var_true = (
            tt_var_true.to(**self.torch_params) if tt_var_true is not None else None
        )
        self.q_prior = (
            to_device(q_prior, params["device"]) if q_prior is not None else None
        )
        self.demand_df = demand_df.copy() if demand_df is not None else None
        self.obs_lids_all = [int(x) for x in obs_lids]
        self.obs_lf_true_all = (
            obs_lf_true.to(**self.torch_params)
            if obs_lf_true is not None
            else self.obs_lf_sample.median(dim=0)[0]
        )
        self.obs_lids = self.obs_lids_all
        self.obs_lf_true = self.obs_lf_true_all
        self.train_obs_lids = self.obs_lids_all
        self.test_obs_lids = []

    def true_demand(self, net):
        if self.demand_df is None:
            return None
        lookup = {
            (int(row.origin), int(row.dest)): float(row.demand)
            for row in self.demand_df.itertuples(index=False)
        }
        out = torch.zeros(len(net.od2did), **self.torch_params)
        for (od, did) in net.od2did.items():
            if od not in lookup:
                raise KeyError(f"OD pair {od} is absent from demand.csv.")
            out[did] = lookup[od]
        return out

    def mean_full_flow(self):
        if self.lf_true is None:
            return None
        lf = self.lf_true
        if lf.dim() == 1:
            return lf
        if lf.dim() == 2:
            return lf.mean(dim=0)
        raise ValueError(f"Unexpected lf_true shape: {tuple(lf.shape)}")

    def flow_lid_groups(self, num_links: int):
        obs = sorted([int(x) for x in self.obs_lids_all])
        obs_set = set(obs)
        unobs = [lid for lid in range(num_links) if lid not in obs_set]
        full = list(range(num_links))
        return (obs, unobs, full)

    def cal_beta(self, net, traj_lids):
        obs_lids = self.obs_lids
        obs_lid_dict = {lid: idx for (idx, lid) in enumerate(obs_lids)}
        obs_traj_lf = torch.zeros(len(obs_lids), **net.torch_params)
        for traj in traj_lids:
            for lid in traj:
                if lid in obs_lid_dict:
                    obs_traj_lf[obs_lid_dict[lid]] += 1
        self.obs_traj_lf = obs_traj_lf
        sample = self.obs_lf_sample
        ratio = (obs_traj_lf / sample).flatten()
        ratio = ratio[torch.isfinite(ratio)]
        self.beta = torch.mean(ratio)
        self.beta = torch.clamp(
            self.beta,
            min=torch.tensor(0.0001, device=self.beta.device, dtype=self.beta.dtype),
            max=torch.tensor(0.95, device=self.beta.device, dtype=self.beta.dtype),
        )

    def gen_traj_qd(self, net, traj):
        if not hasattr(traj, "q"):
            traj.gen_traj_info(net)
        if traj.q.numel() != len(net.od2did):
            raise ValueError(
                "Trajectory OD counts are not aligned to the 528-row demand table."
            )
        if bool((traj.q <= 0).any()):
            missing = torch.nonzero(traj.q <= 0, as_tuple=False).flatten().tolist()
            raise ValueError(
                f"Every OD must have at least one training trajectory; zero-count dids={missing}. Regenerate the OD-wise train/test split."
            )
        self.traj_q = traj.q.clone()

    def cal_l2(self, obs_lf, obs_lf_cov=None):
        if obs_lf_cov is None:
            raise ValueError(
                "Sioux Falls Gaussian flow loss requires a covariance matrix."
            )
        if not bool(torch.isfinite(obs_lf).all()):
            raise FloatingPointError(
                "Predicted observed-link flow is non-finite before the Gaussian flow loss. Check the predicted flow moments."
            )
        if not bool(torch.isfinite(obs_lf_cov).all()):
            raise FloatingPointError(
                "Predicted observed-flow covariance is non-finite before the Gaussian flow loss. Check the predicted flow moments."
            )
        sample = self.obs_lf_sample.to(device=obs_lf.device, dtype=obs_lf.dtype)
        (num_days, num_obs_links) = sample.shape
        eye = torch.eye(num_obs_links, device=obs_lf.device, dtype=obs_lf.dtype)
        obs_lf_cov = obs_lf_cov + 1e-06 * eye
        (_, log_det) = torch.linalg.slogdet(obs_lf_cov)
        obs_lf_cov_inv = torch.linalg.inv(obs_lf_cov)
        residual = sample - obs_lf.unsqueeze(0)
        quadratic = torch.sum(residual * (residual @ obs_lf_cov_inv.T))
        return 0.5 * (quadratic + num_days * log_det)

    def cal_l3_components(self, gamma, params):
        g = gamma["od"]
        beta = self.beta.to(device=g.device, dtype=g.dtype)
        alpha_loss = torch.sum((gamma2alpha(g) - beta) ** 2)
        od_loss = torch.zeros_like(alpha_loss)
        if self.q_prior is None:
            raise RuntimeError("OD prior is enabled but q_prior was not loaded.")
        q_prior = self.q_prior.to(device=g.device, dtype=g.dtype)
        if q_prior.numel() != g.numel():
            raise ValueError(
                f"q_prior must have {g.numel()} OD entries, got {q_prior.numel()}."
            )
        traj_q = self.traj_q.to(device=g.device, dtype=g.dtype)
        od_loss = torch.sum((q_prior - traj_q / gamma2alpha(g)) ** 2)
        return {
            "alpha": params["lambda_alpha"] * alpha_loss,
            "od": params["lambda_od"] * od_loss,
        }


def load_flow(params: dict) -> FlowSample:
    root = Path(params["data_root"]) / "flow"
    folder = root
    params["flow_folder"] = str(folder)
    obs_lf_sample = torch.load(
        folder / "obs_lf_sample.pt", map_location=params["device"], weights_only=True
    )
    lf_true = torch.load(
        folder / "lf_true.pt", map_location=params["device"], weights_only=True
    )
    lf_cov_true = torch.load(
        folder / "lf_cov_true.pt", map_location=params["device"], weights_only=True
    )
    state_feature_true = torch.load(
        Path(params["data_root"]) / "net" / "state_feature.pt",
        map_location=params["device"],
        weights_only=True,
    )
    with open(folder / "obs_lids.json", "r", encoding="utf-8") as f:
        obs_lids = json.load(f)
    q_prior_path = folder / "q_prior.pt"
    if not q_prior_path.exists():
        raise FileNotFoundError(
            f"OD prior is enabled but {q_prior_path} does not exist. The bundled flow data must include q_prior.pt."
        )
    params["q_prior_path"] = str(q_prior_path)
    q_prior = (
        torch.load(q_prior_path, map_location=params["device"], weights_only=True)
        if q_prior_path.exists()
        else None
    )
    demand_true_path = Path(params["data_root"]) / "net" / "demand.csv"
    params["demand_true_path"] = str(demand_true_path)
    demand_df = pd.read_csv(demand_true_path)[["origin", "dest", "demand"]]
    return FlowSample(
        obs_lf_sample,
        lf_true,
        obs_lids,
        params,
        lf_cov_true=lf_cov_true,
        tt_true=state_feature_true[: len(lf_true), 0],
        tt_var_true=state_feature_true[: len(lf_true), 1],
        q_prior=q_prior,
        demand_df=demand_df,
    )
