"""Fine-tune TimesFM 2.5 (200M) on next-day realised variance, pooled across the 20 indices (run with .venv-tfm, GPU).

The "fine-tuned" (FT) arm of Goel et al. (2025), simplified: one pooled model instead of one per index per round,
trained once on data strictly before the earliest test date of any index (CUTOFF = 2013-05-03, the start of AORD's
test half), minus an H = 21-day embargo, so every index's test region is out of sample.
Training samples: windows of L = 512 days ending before CUTOFF, in levels or in logs with equal probability (the
model is then used for both TFM* and TFM*log style forecasts). Loss, in the same per-window normalised space the
library uses at inference (global mean/std of the context, then RevIN per patch, flip-invariant average):
    MSE of the mean head on the target  +  pinball loss of the nine decile heads
with the normalised target clipped to [-10, 30] so that a single vol spike cannot dominate a batch.
Hyperparameters are fixed a priori (full fine-tune, AdamW lr 1e-5, wd 0.01, batch 64, bf16, 4000 steps, 200 warmup,
cosine). The only data-driven choice is early stopping on the last 252 trading days before the embargo (validation
set, also before CUTOFF), checked every 250 steps.
Writes models/timesfm_ft.safetensors (gitignored, ~0.9 GB) and models/timesfm_ft_log.json (loss curve).
Usage: .venv-tfm/bin/python timesfm_finetune.py [--steps N]
"""
import json
import math
import os
import sys
import time
import numpy as np
import pandas as pd
import timesfm
import torch
from safetensors.torch import save_file
from timesfm.torch.util import revin, update_running_stats

L, BATCH, LR, WD, WARMUP, EVAL_EVERY, H = 512, 64, 1e-5, 0.01, 200, 250, 21
STEPS = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 4000
QUANTILES = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
CUTOFF = pd.Timestamp("2013-05-03")
torch.manual_seed(0); rng = np.random.default_rng(0)

rv = pd.read_csv("data/rv_daily.csv", index_col=0, parse_dates=True)
series, train_idx, val_idx = [], [], []                     # (series id, target position) pairs
for name in rv.columns:
    s = rv[name].dropna()
    assert s.index[len(s) // 2] >= CUTOFF, name             # CUTOFF must precede every test region
    last = np.searchsorted(s.index, CUTOFF) - H             # last allowed target position (exclusive)
    for log in (False, True):
        k = len(series); series.append((np.log(s.values) if log else s.values).astype(np.float32))
        pos = np.arange(L, last)
        train_idx += [(k, t) for t in pos[pos < last - 252 - H]]
        val_idx += [(k, t) for t in pos[pos >= last - 252]]
train_idx, val_idx = np.array(train_idx), np.array(val_idx)
print(f"{len(train_idx)} training windows, {len(val_idx)} validation windows", flush=True)

wrapper = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
net = wrapper.model; dev = net.device; net.train()
qs = QUANTILES.to(dev)


def batch(idx):
    x = np.stack([series[k][t - L:t] for k, t in idx]); y = np.array([series[k][t] for k, t in idx])
    return torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev)


def core(z):
    """Differentiable prefill of TimesFM_2p5_200M_torch_module.decode: next-step outputs, all 10 heads."""
    b = z.shape[0]
    pz = z.reshape(b, -1, net.p); pm = torch.zeros_like(pz, dtype=torch.bool)
    n = mu = sg = torch.zeros(b, device=dev)
    mus, sgs = [], []
    for i in range(pz.shape[1]):
        (n, mu, sg), _ = update_running_stats(n, mu, sg, pz[:, i], pm[:, i])
        mus.append(mu); sgs.append(sg)
    cmu, csg = torch.stack(mus, 1), torch.stack(sgs, 1)
    (_, _, out, _), _ = net(revin(pz, cmu, csg), pm, None)
    out = revin(out, cmu, csg, reverse=True).reshape(b, -1, net.o, net.q)
    return out[:, -1, 0, :]


def loss_fn(x, y):
    mu = x.mean(-1, keepdim=True); sg = x.std(-1, keepdim=True).clamp_min(1e-8)
    xn = (x - mu) / sg; yn = ((y[:, None] - mu) / sg).clamp(-10, 30)[:, 0]
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        f, g = core(xn), core(-xn)
    f, g = f.float(), g.float()
    out = (f - torch.cat([g[:, :1], torch.flip(g[:, 1:], dims=(-1,))], -1)) / 2   # library's flip invariance
    e = yn[:, None] - out[:, 1:]
    return ((out[:, 0] - yn) ** 2).mean() + torch.maximum(qs * e, (qs - 1) * e).mean() * 2


@torch.no_grad()
def val_loss():
    net.eval()
    tot = [loss_fn(*batch(val_idx[i:i + 256])).item() * len(val_idx[i:i + 256]) for i in range(0, len(val_idx), 256)]
    net.train()
    return sum(tot) / len(val_idx)


opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1, (s + 1) / WARMUP) * 0.5 * (1 + math.cos(math.pi * min(s, STEPS) / STEPS)))
os.makedirs("models", exist_ok=True)
best, log, t0, run = val_loss(), [], time.time(), 0.0
log.append({"step": 0, "val": best}); print(f"step 0 val {best:.4f} (zero-shot)", flush=True)
save_file({k: v.detach().cpu().contiguous() for k, v in net.state_dict().items()}, "models/timesfm_ft.safetensors")
for step in range(1, STEPS + 1):
    loss = loss_fn(*batch(train_idx[rng.integers(len(train_idx), size=BATCH)]))
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step(); sched.step(); run += loss.item()
    if step % EVAL_EVERY == 0:
        v = val_loss(); log.append({"step": step, "train": run / EVAL_EVERY, "val": v}); run = 0.0
        tag = ""
        if v < best:
            best, tag = v, "  saved"
            save_file({k: w.detach().cpu().contiguous() for k, w in net.state_dict().items()},
                      "models/timesfm_ft.safetensors")
        print(f"step {step} train {log[-1]['train']:.4f} val {v:.4f} ({time.time() - t0:.0f}s){tag}", flush=True)
json.dump({"cutoff": str(CUTOFF.date()), "steps": STEPS, "lr": LR, "batch": BATCH, "best_val": best, "curve": log},
          open("models/timesfm_ft_log.json", "w"), indent=1)
