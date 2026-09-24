from __future__ import annotations
import math
from pathlib import Path
from typing import Dict
import networkx as nx
import numpy as np
import pandas as pd
import torch


class Net:
    def __init__(self, link_df: pd.DataFrame, demand_df: pd.DataFrame, params: dict):
        self.torch_params = {
            "device": params["device"],
            "dtype": params["tensor_dtype"],
        }
        link_df = link_df.copy()
        demand_df = (
            demand_df[["origin", "dest", "demand"]].copy().reset_index(drop=True)
        )
        (from_col, to_col) = ("from", "to")
        (len_col, cap_col, fft_col) = ("len", "cap", "fft")
        self.from_col = from_col
        self.to_col = to_col
        self.L = link_df.shape[0]
        self.num_real_links = self.L
        self.lids = list(range(self.L))
        self.real_lids = list(self.lids)
        link_df["lid"] = self.lids
        self.nids = list(set(link_df[from_col].tolist() + link_df[to_col].tolist()))
        self.ft2lid = dict(zip(zip(link_df[from_col], link_df[to_col]), self.lids))
        self.lid2ft = dict(zip(self.lids, zip(link_df[from_col], link_df[to_col])))
        self.nid2upstrm = link_df.groupby(to_col)["lid"].apply(list).to_dict()
        self.nid2dstrm = link_df.groupby(from_col)["lid"].apply(list).to_dict()
        self.link_df = link_df
        self.demand_df = demand_df
        self.origins = sorted((int(x) for x in demand_df.origin.unique()))
        self.dests = sorted((int(x) for x in demand_df.dest.unique()))
        self.od2did = {
            (int(row.origin), int(row.dest)): int(index)
            for (index, row) in demand_df.iterrows()
        }
        self.did2od = {did: od for (od, did) in self.od2did.items()}
        if len(self.od2did) != len(demand_df):
            raise ValueError("demand.csv contains duplicate OD pairs.")
        self.origin_virtual_lid = {
            origin: self.L + index for (index, origin) in enumerate(self.origins)
        }
        self.virtual_lid2origin = {
            virtual_lid: origin
            for (origin, virtual_lid) in self.origin_virtual_lid.items()
        }
        self.virtual_origin_lids = list(self.origin_virtual_lid.values())
        self.terminal_lid = self.L + len(self.origins)
        self.num_choice_states = self.terminal_lid + 1

        def tensor_col(col, default):
            if col is None or col not in link_df.columns:
                return torch.full((self.L,), float(default), **self.torch_params)
            return torch.tensor(link_df[col].astype(float).values, **self.torch_params)

        self.len = tensor_col(len_col, 1.0)
        self.cap = tensor_col(cap_col, 10000.0)
        self.fft = tensor_col(fft_col, 1.0)
        self.t = self.fft.clone()

    def add_demand_info(self, traj_demand: pd.DataFrame):
        lookup = {
            (int(row.origin), int(row.dest)): float(row.demand)
            for row in traj_demand.itertuples(index=False)
        }
        self.traj_demand_df = self.demand_df[["origin", "dest"]].copy()
        self.traj_demand_df["demand"] = [
            lookup.get((int(row.origin), int(row.dest)), 0.0)
            for row in self.traj_demand_df.itertuples(index=False)
        ]
        self.traj_demand = torch.tensor(
            self.traj_demand_df["demand"].values, **self.torch_params
        )

    def od_vector_to_state_demand(
        self, demand: torch.Tensor
    ) -> Dict[int, torch.Tensor]:
        if demand.numel() != len(self.od2did):
            raise ValueError(
                f"OD demand must have {len(self.od2did)} entries, got {demand.numel()}."
            )
        out = {
            dest: torch.zeros(
                self.num_choice_states, device=demand.device, dtype=demand.dtype
            )
            for dest in self.dests
        }
        for ((origin, dest), did) in self.od2did.items():
            out[dest][self.origin_virtual_lid[origin]] = demand[did]
        return out

    def build_dual_graph(self):
        left_to = self.to_col
        right_from = self.from_col
        real_dual_df = self.link_df.merge(
            self.link_df,
            left_on=left_to,
            right_on=right_from,
            suffixes=("_curr", "_next"),
        )
        real_dual_df = real_dual_df[
            real_dual_df["lid_curr"] != real_dual_df["lid_next"]
        ].reset_index(drop=True)
        real_pairs = list(
            zip(
                real_dual_df["lid_curr"].astype(int),
                real_dual_df["lid_next"].astype(int),
            )
        )
        origin_pairs = [
            (self.origin_virtual_lid[origin], int(real_lid))
            for origin in self.origins
            for real_lid in self.nid2dstrm[origin]
        ]
        pairs = real_pairs + origin_pairs
        self.num_real_tids = len(real_pairs)
        self.tids = list(range(len(pairs)))
        self.tid2ft = dict(zip(self.tids, pairs))
        self.ft2tid = dict(zip(pairs, self.tids))
        self.origin_transition_tids = {
            (origin, int(real_lid)): self.ft2tid[
                self.origin_virtual_lid[origin], int(real_lid)
            ]
            for origin in self.origins
            for real_lid in self.nid2dstrm[origin]
        }
        (self.lid2dstrm, self.lid2upstrm) = ({}, {})
        for (from_lid, to_lid) in pairs:
            self.lid2dstrm.setdefault(from_lid, []).append(to_lid)
            self.lid2upstrm.setdefault(to_lid, []).append(from_lid)
        self.tail_lids = [pair[0] for pair in pairs]
        self.head_lids = [pair[1] for pair in pairs]
        self.dual_link_df = real_dual_df

    def cal_edge_index(self):
        head = np.asarray(self.head_lids, dtype=np.int64)
        tail = np.asarray(self.tail_lids, dtype=np.int64)
        self.edge_index = (
            torch.from_numpy(np.stack([head, tail], axis=0))
            .long()
            .to(self.torch_params["device"])
        )

    def cal_shortest_step(self):
        G = nx.DiGraph()
        G.add_edges_from(
            [(self.lid2ft[lid][1], self.lid2ft[lid][0]) for lid in self.lids]
        )
        self.dest_mini_steps = {
            d: nx.single_source_shortest_path_length(G, d) for d in self.dests
        }

    def cal_step(self, traj, params):
        dest_spt_steps = self.dest_mini_steps
        self.dest_steps = {d: {} for d in self.dests}
        for lid in self.lids:
            to_nid = self.lid2ft[lid][1]
            for d in self.dests:
                if to_nid in dest_spt_steps[d]:
                    self.dest_steps[d][lid] = dest_spt_steps[d][to_nid] + 1
        zeta = float(params.get("prism_elongation_ratio", 1.0))
        if zeta < 0:
            raise ValueError("The prism elongation ratio zeta must be nonnegative.")
        shortest_choice_stages = {
            d: max(
                (
                    int(dest_spt_steps[d][int(origin)]) + 1
                    for (origin, dest) in self.od2did
                    if int(dest) == int(d)
                )
            )
            for d in self.dests
        }
        base_stages = shortest_choice_stages
        self.T = {d: int(math.ceil((1.0 + zeta) * base_stages[d])) for d in self.dests}
        self.prism_base_stages = base_stages
        self.prism_elongation_ratio = zeta
        self.prism_horizon_basis = "shortest_path"


def load_net(params: dict):
    root = Path(params["data_root"]) / "net"
    link_df = pd.read_csv(root / "Links.csv")
    demand_df = pd.read_csv(root / "demand.csv")
    net = Net(link_df, demand_df, params)
    net.build_dual_graph()
    return net
