from pathlib import Path

import torch


def _settings():
    root = Path(__file__).resolve().parents[1]
    params = {
        "data_root": str(root / "data"),
        "results_root": str(root / "results"),
        "device": "cpu",
        "tensor_dtype": torch.float32,
        "seed": 42,
        "deterministic": True,
        "prism_elongation_ratio": 1.0,
        "batch_size": 4096,
        "test_size": 0.2,
        "lambda_alpha": 100.0,
        "lambda_od": 0.001,
        "lambda_theta": 1.0,
        "theta_lr": 1e-2,
        "gamma_lr": 20.0,
        "num_epoch": 1000,
        "early_stop": True,
        "early_stop_tol": 5e-4,
        "early_stop_min_epoch": 150,
        "early_stop_patience": 4,
        "batch_norm_track_running_stats": False,
    }
    reward_params = {
        "state_hidden_dim": 16,
        "action_hidden_dim": 16,
        "state_out_dim": 8,
        "action_out_dim": 8,
        "trans_hidden_dim": 8,
        "heads": 2,
        "dropout": 0,
    }
    return params, reward_params
