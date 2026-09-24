from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch


class TrajSample:
    def __init__(self, traj_trans: list, traj_lids: list, traj_dests: list):
        self.traj_trans = traj_trans
        self.traj_lids = traj_lids
        self.traj_dests = traj_dests
        self.len_trajs = len(traj_lids)

    def gen_traj_od_df(self, net) -> pd.DataFrame:
        od_tab = {}
        for (trans, lids, dest) in zip(
            self.traj_trans, self.traj_lids, self.traj_dests
        ):
            if lids and trans:
                virtual_lid = net.tid2ft[int(trans[0])][0]
                origin = net.virtual_lid2origin[int(virtual_lid)]
                od_tab[origin, dest] = od_tab.get((origin, dest), 0) + 1
        rows = [(o, d, n) for ((o, d), n) in od_tab.items()]
        return (
            pd.DataFrame(rows, columns=["origin", "dest", "demand"])
            .sort_values(["origin", "dest"])
            .reset_index(drop=True)
        )

    def gen_traj_info(self, net):
        q = torch.zeros(len(net.od2did), **net.torch_params)
        rho_real = torch.zeros(len(net.tids), **net.torch_params)
        psi_real = torch.zeros(net.num_real_links, **net.torch_params)
        traj_steps = {d: {} for d in net.dests}
        device = net.torch_params["device"]
        for (trans, lids, dest) in zip(
            self.traj_trans, self.traj_lids, self.traj_dests
        ):
            if not lids:
                continue
            if not trans:
                raise ValueError(
                    "Every Sioux Falls trajectory must start with a virtual-origin transition."
                )
            virtual_lid = net.tid2ft[int(trans[0])][0]
            origin = net.virtual_lid2origin.get(int(virtual_lid))
            if origin is None:
                raise ValueError(
                    f"Trajectory does not start at a virtual origin: transition={trans[0]}."
                )
            q[net.od2did[origin, int(dest)]] += 1
            if trans:
                rho_real[torch.tensor(trans, device=device, dtype=torch.long)] += 1
            psi_real[torch.tensor(lids, device=device, dtype=torch.long)] += 1
            for (step, tid) in enumerate(trans):
                (from_lid, to_lid) = net.tid2ft[tid]
                traj_steps.setdefault(dest, {}).setdefault(step, [[], []])
                traj_steps[dest][step][0].append(from_lid)
                traj_steps[dest][step][1].append(to_lid)
            final_step = len(trans)
            traj_steps.setdefault(dest, {}).setdefault(final_step, [[], []])
            traj_steps[dest][final_step][0].append(lids[-1])
            traj_steps[dest][final_step][1].append(net.terminal_lid)
        self.q = q
        self.qd = net.od_vector_to_state_demand(q)
        self.rho_real = rho_real
        self.psi_real = psi_real
        self.traj_steps = traj_steps

    def gen_od2traj_id(self):
        out = {}
        for (traj_id, (trans, lids)) in enumerate(zip(self.traj_trans, self.traj_lids)):
            if lids and trans:
                dest = self.traj_dests[traj_id]
                start_virtual_lid = self._start_virtual_lid_by_transition[int(trans[0])]
                out.setdefault(dest, {}).setdefault(start_virtual_lid, []).append(
                    traj_id
                )
        return out

    def bind_network(self, net):
        self._start_virtual_lid_by_transition = {
            tid: from_lid
            for (tid, (from_lid, _)) in net.tid2ft.items()
            if from_lid in net.virtual_lid2origin
        }


def _traj_folder(params, true=False):
    root = Path(params["data_root"]) / "traj"
    return root / "true_traj" if true else root


def load_traj(params: dict, true=False) -> TrajSample:
    folder = _traj_folder(params, true)
    with open(folder / "traj_trans.json", "r", encoding="utf-8") as f:
        traj_trans = json.load(f)
    with open(folder / "traj_lids.json", "r", encoding="utf-8") as f:
        traj_lids = json.load(f)
    with open(folder / "traj_dests.json", "r", encoding="utf-8") as f:
        traj_dests = json.load(f)
    if not len(traj_trans) == len(traj_lids) == len(traj_dests):
        raise ValueError(f"Inconsistent trajectory file lengths under {folder}.")
    return TrajSample(traj_trans, traj_lids, traj_dests)


def _subset(traj: TrajSample, indices) -> TrajSample:
    return TrajSample(
        [traj.traj_trans[i] for i in indices],
        [traj.traj_lids[i] for i in indices],
        [traj.traj_dests[i] for i in indices],
    )


def _new_split_indices(traj: TrajSample, net, params: dict):
    groups = {}
    for (index, (trans, dest)) in enumerate(zip(traj.traj_trans, traj.traj_dests)):
        if not trans:
            raise ValueError(f"Trajectory {index} has no virtual-origin transition.")
        virtual_lid = net.tid2ft[int(trans[0])][0]
        origin = net.virtual_lid2origin.get(int(virtual_lid))
        if origin is None:
            raise ValueError(f"Trajectory {index} does not begin at a virtual origin.")
        groups.setdefault((origin, int(dest)), []).append(index)
    rng = np.random.default_rng(params["seed"])
    (train_idx, test_idx) = ([], [])
    for od in sorted(groups):
        indices = np.asarray(groups[od], dtype=int)
        indices = rng.permutation(indices)
        if len(indices) == 1:
            n_test = 0
        else:
            n_test = int(round(len(indices) * float(params["test_size"])))
            n_test = min(max(n_test, 1), len(indices) - 1)
        test_idx.extend((int(i) for i in indices[:n_test]))
        train_idx.extend((int(i) for i in indices[n_test:]))
    return (sorted(train_idx), sorted(test_idx))


def _load_or_create_split(traj: TrajSample, folder: Path, net, params: dict):
    train_path = folder / "train_indices.json"
    test_path = folder / "test_indices.json"
    (expected_train_idx, expected_test_idx) = _new_split_indices(traj, net, params)
    if train_path.exists() and test_path.exists():
        with open(train_path, "r", encoding="utf-8") as f:
            train_idx = json.load(f)
        with open(test_path, "r", encoding="utf-8") as f:
            test_idx = json.load(f)
        if train_idx != expected_train_idx or test_idx != expected_test_idx:
            (train_idx, test_idx) = (expected_train_idx, expected_test_idx)
    else:
        (train_idx, test_idx) = (expected_train_idx, expected_test_idx)
    combined = train_idx + test_idx
    if len(combined) != traj.len_trajs or sorted(combined) != list(
        range(traj.len_trajs)
    ):
        raise ValueError(
            f"Saved train/test indices under {folder} do not match the current trajectory files. Remove the two index files only after intentionally replacing the data."
        )
    return (train_idx, test_idx)


def load_train_test(params, net):
    traj_all = load_traj(params)
    unbiased = load_traj(params, true=True)
    (_, unbiased_test_idx) = _load_or_create_split(
        unbiased, _traj_folder(params, true=True), net, params
    )
    (train_idx, test_idx) = _load_or_create_split(
        traj_all, _traj_folder(params), net, params
    )
    values = (
        traj_all,
        _subset(traj_all, train_idx),
        _subset(traj_all, test_idx),
        _subset(unbiased, unbiased_test_idx),
    )
    for value in values:
        value.bind_network(net)
    return values
