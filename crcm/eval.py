from __future__ import annotations
from collections import Counter
import numpy as np
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from scipy.optimize import linear_sum_assignment

try:
    import editdistance
except Exception:

    class _EditDistanceFallback:
        @staticmethod
        def eval(a, b):
            a = list(a)
            b = list(b)
            (m, n) = (len(a), len(b))
            dp = list(range(n + 1))
            for i in range(1, m + 1):
                (prev, dp[0]) = (dp[0], i)
                for j in range(1, n + 1):
                    cur = dp[j]
                    cost = 0 if a[i - 1] == b[j - 1] else 1
                    dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
                    prev = cur
            return dp[n]

    editdistance = _EditDistanceFallback()


def cal_r_squared(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    y_true = y_true.to(y_pred.device, dtype=y_pred.dtype)
    y_pred = y_pred.to(y_true.device, dtype=y_true.dtype)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


def cal_mape(
    y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-10
) -> torch.Tensor:
    y_true = y_true.to(y_pred.device, dtype=y_pred.dtype)
    y_pred = y_pred.to(y_true.device, dtype=y_true.dtype)
    return torch.mean(torch.abs((y_true - y_pred) / (y_true + eps)))


def gen_prob_path(net, RL, od2traj_id, num_path, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    learn_path_dict = {}
    for d in od2traj_id.keys():
        P_d_obj = RL.p_mat[d]
        pos_dest = net.terminal_lid
        learn_path_dict[d] = {}
        for start_link in od2traj_id[d].keys():
            n = len(od2traj_id[d][start_link]) if num_path == -1 else int(num_path)
            if n < 0:
                raise ValueError("num_path must be -1 or a non-negative integer.")
            generated = []
            for sample_id in range(n):
                path = []
                cur_link = int(start_link)
                max_steps = len(P_d_obj)
                for t in range(max_steps):
                    if cur_link == pos_dest:
                        break
                    P_obj = P_d_obj[t]
                    P_dt = P_obj.detach().to_dense().cpu().numpy()
                    next_link_prob = P_dt[cur_link].copy()
                    next_link_prob[cur_link] = 0
                    next_link_prob = np.clip(next_link_prob, 0.0, None)
                    prob_sum = next_link_prob.sum()
                    if not np.isfinite(prob_sum) or prob_sum <= 0:
                        raise RuntimeError(
                            f"Path generation found no valid transition for destination={d}, virtual_origin={start_link}, sample={sample_id}, time_step={t}, state={cur_link}."
                        )
                    next_link = rng.choice(
                        np.arange(net.num_choice_states), p=next_link_prob / prob_sum
                    )
                    cur_link = int(next_link)
                    if cur_link < net.num_real_links:
                        path.append(cur_link)
                if cur_link != pos_dest:
                    raise RuntimeError(
                        f"Path generation did not reach the terminal state within the {'prism horizon'} for destination={d}, virtual_origin={start_link}, sample={sample_id}."
                    )
                generated.append(path)
            learn_path_dict[d][start_link] = generated
    return learn_path_dict


def _od_matched_path_pairs(traj_test, od2traj_id, learn_path_dict):
    traj_lids = traj_test.traj_lids
    for (d, start_groups) in od2traj_id.items():
        for (start_link, traj_ids) in start_groups.items():
            references = [[int(x) for x in traj_lids[i]] for i in traj_ids]
            predictions = [
                [int(x) for x in path]
                for path in learn_path_dict.get(d, {}).get(start_link, [])
            ]
            if len(predictions) != len(traj_ids):
                raise ValueError(
                    f"Generated-path count must equal the observed trajectory count for destination={d}, virtual_origin={start_link}: generated={len(predictions)}, observed={len(traj_ids)}."
                )
            if not references:
                continue
            cost = np.empty((len(references), len(predictions)), dtype=float)
            for (i, reference) in enumerate(references):
                for (j, prediction) in enumerate(predictions):
                    denominator = max(len(reference), len(prediction), 1)
                    cost[i, j] = min(
                        editdistance.eval(reference, prediction) / denominator, 1.0
                    )
            (reference_ids, prediction_ids) = linear_sum_assignment(cost)
            for (i, j) in zip(reference_ids.tolist(), prediction_ids.tolist()):
                yield (references[i], predictions[j], float(cost[i, j]))


def cal_path_metrics(net, traj_test, od2traj_id, learn_path_dict):
    smoothie = SmoothingFunction().method2
    bleu_scores = []
    edit_distances = []
    overlap_scores = []
    link_lengths = {
        lid: float(net.len[lid].detach().cpu()) for lid in range(net.num_real_links)
    }
    for (reference, prediction, edit_distance) in _od_matched_path_pairs(
        traj_test, od2traj_id, learn_path_dict
    ):
        bleu_scores.append(
            sentence_bleu(
                [reference], prediction, smoothing_function=smoothie, weights=(0.5, 0.5)
            )
        )
        edit_distances.append(edit_distance)
        ref_counts = Counter(reference)
        pred_counts = Counter(prediction)
        ref_length = sum(
            (count * link_lengths[lid] for (lid, count) in ref_counts.items())
        )
        pred_length = sum(
            (count * link_lengths[lid] for (lid, count) in pred_counts.items())
        )
        common_counts = ref_counts & pred_counts
        common_length = sum(
            (count * link_lengths[lid] for (lid, count) in common_counts.items())
        )
        denominator = ref_length + pred_length
        overlap_scores.append(
            2.0 * common_length / denominator if denominator > 0 else 1.0
        )
    return {
        "BLEU": float(np.mean(bleu_scores)) if bleu_scores else None,
        "ED": float(np.mean(edit_distances)) if edit_distances else None,
        "PSS": float(np.mean(overlap_scores)) if overlap_scores else None,
    }
