# Consistent Route Choice Model

Python code for the Consistent Route Choice Model (CRCM): integrating individual and aggregate traffic data.

CRCM combines GPS trajectories and link flow observations within a likelihood-based framework to jointly estimate route choice preferences and network flows. Prism-constrained deep inverse reinforcement learning captures nonlinear feature relationships and latent correlations among choice alternatives while limiting unreasonable detours.

## Environment

Python with PyTorch, PyTorch Geometric, NumPy, pandas, SciPy, NetworkX, NLTK, and editdistance.

```bash
pip install -r requirements.txt
python crcm_main.py
```

The entry point trains the model and then evaluates it using the included data.

## Files

- `crcm/`: reward model, prism loading, training, and evaluation.
- `data/net/`: network topology, OD demand, and input features.
- `data/flow/`: link flow observations, reference moments, and the OD prior.
- `data/traj/`: trajectories and training/test split indices.
- `data/traj/true_traj/`: independent reference trajectories and split indices; evaluation uses the held-out test subset.

Outputs are saved to `results/`: reward weights (`reward.pth`), effective settings (`config.json`), training history (`iteration.json`), evaluation metrics (`res.json`), link flows (`flow.csv`), OD demand (`qd.csv`), and supplied travel-time moments (`travel_time.csv`).

Trajectory evaluation reports average negative log-likelihood, aggregate link-count R², BLEU, normalized edit distance, and path similarity for `train_traj`, `test_traj`, and `true_traj`. Flow evaluation reports mean and variance R² and MAPE for observed links (`train_flow`), unobserved links (`test_flow`), and all links (`full_flow`). OD demand is evaluated separately. Travel-time metrics compare the supplied moments and do not measure an estimated congestion response.
