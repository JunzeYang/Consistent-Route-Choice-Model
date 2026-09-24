import json
import os
import random
from pathlib import Path
from typing import Any, Optional
import numpy as np
import pandas as pd
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def to_device(obj: Any, device: str) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for (k, v) in obj.items()}
    if isinstance(obj, list):
        return [to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple((to_device(v, device) for v in obj))
    return obj


def write_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, ensure_ascii=False, indent=2)


def experiment_dir(params):
    return Path(params["results_root"])


def result_path(params, content):
    return experiment_dir(params) / content


def save_flow_csv(
    flow: torch.Tensor, path: str, flow_variance: Optional[torch.Tensor] = None
) -> None:
    flow = flow.detach().cpu().flatten().numpy()
    data = {"lid": np.arange(len(flow), dtype=int), "flow_mean": flow}
    if flow_variance is not None:
        data["flow_variance"] = flow_variance.detach().cpu().flatten().numpy()
    pd.DataFrame(data).to_csv(path, index=False)


def save_travel_time_csv(mean: torch.Tensor, variance: torch.Tensor, path: str) -> None:
    mean = mean.detach().cpu().flatten().numpy()
    variance = variance.detach().cpu().flatten().numpy()
    pd.DataFrame(
        {
            "lid": np.arange(len(mean), dtype=int),
            "travel_time_mean": mean,
            "travel_time_variance": variance,
        }
    ).to_csv(path, index=False)


def save_qd_csv(net: Any, demand: torch.Tensor, path: str) -> None:
    demand = demand.detach().cpu().flatten().numpy()
    if len(demand) != len(net.od2did):
        raise ValueError(
            f"qd output must contain {len(net.od2did)} OD entries, got {len(demand)}."
        )
    rows = []
    for ((origin, dest), did) in sorted(net.od2did.items(), key=lambda item: item[1]):
        rows.append(
            {
                "did": int(did),
                "origin": int(origin),
                "dest": int(dest),
                "virtual_lid": int(net.origin_virtual_lid[origin]),
                "demand": float(demand[did]),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def to_jsonable(obj):
    import numpy as np
    import torch

    if isinstance(obj, dict):
        return {k: to_jsonable(v) for (k, v) in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.detach().cpu().item()
        return obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (torch.dtype, torch.device)):
        return str(obj)
    return obj
