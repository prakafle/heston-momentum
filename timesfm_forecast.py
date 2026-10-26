"""Zero-shot TimesFM forecasts of next-day realised variance (run with .venv-tfm/bin/python, Python 3.11).

Replicates the "pre-trained" (PT) arm of Goel et al. (2025): context lengths 64, 128, 512, horizon 1, on RV and on
log RV (their superscript "log" variants). We use TimesFM 2.5 (200M), the successor of the 2.0-500M checkpoint the
paper used; the paper's GPU fine-tuning (40k steps per index per round) is not reproduced here.
For every index and every day t in the second half of its sample, the model sees RV_{t-L..t-1} and predicts RV_t.
Writes data/timesfm_forecasts.csv: date, index, variant, forecast (daily variance).
Usage: .venv-tfm/bin/python timesfm_forecast.py [--quick] [--cpu]   (uses the Apple GPU via MPS when present)
"""
import sys
import time
import numpy as np
import pandas as pd
import timesfm
import torch

CONTEXTS = [64, 128, 512]
BATCH = 256
quick = "--quick" in sys.argv

torch.set_float32_matmul_precision("high")
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
if torch.backends.mps.is_available() and "--cpu" not in sys.argv:   # library only auto-detects CUDA
    model.model.device = torch.device("mps")
    model.model.to(model.model.device)
model.compile(timesfm.ForecastConfig(max_context=512, max_horizon=16, normalize_inputs=True,
                                     per_core_batch_size=BATCH, infer_is_positive=True))
if model.model.device.type == "mps":                  # the library builds float64 arrays internally; MPS needs float32
    _decode = model.compiled_decode
    model.compiled_decode = lambda h, values, masks: _decode(h, np.asarray(values, np.float32), masks)
rv = pd.read_csv("data/rv_daily.csv", index_col=0, parse_dates=True)

rows = []
for name in rv.columns[:1] if quick else rv.columns:
    s = rv[name].dropna()
    start = len(s) // 2                                   # paper: test region = everything after the first 50%
    targets = range(start, start + 300) if quick else range(start, len(s))
    for log in (False, True):
        x = (np.log(s.values) if log else s.values).astype(np.float32)   # MPS has no float64
        for L in CONTEXTS:
            t0 = time.time()
            ctx = [x[t - L:t] for t in targets]
            preds = []
            for i in range(0, len(ctx), BATCH):
                point, _ = model.forecast(horizon=1, inputs=ctx[i:i + BATCH])
                preds.append(point[:, 0])
            p = np.concatenate(preds)
            p = np.exp(p) if log else np.maximum(p, 1e-10)  # log forecasts mapped back without bias correction,
            rows += [{"date": s.index[t], "index": name, "variant": f"TFM{L}{'log' if log else ''}", "forecast": v}
                     for t, v in zip(targets, p)]              # like the paper's (·)^log models
            print(f"{name} L={L} log={log}: {len(p)} forecasts in {time.time() - t0:.1f}s", flush=True)
out = pd.DataFrame(rows)
out.to_csv("data/timesfm_forecasts_quick.csv" if quick else "data/timesfm_forecasts.csv", index=False)
