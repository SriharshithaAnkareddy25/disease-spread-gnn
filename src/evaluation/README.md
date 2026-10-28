# SIR output layer + evaluation (Person 3)

Person 2's model predicts a **beta proxy** — a transmission rate — and stops
there. `training/data.py` builds that target by *inverting* the SIR incidence
equation on the observed case counts, and `train.py` reports error against
that proxy (its own report key is `test_metrics_against_beta_proxy`).

That leaves the pipeline one link short of being usable: nothing converts
beta back into case numbers, so the project could not compare against a
baseline, rank districts by risk, or report anything a health officer could
read.

This module closes that gap and then tests the result honestly.

## What's here

| File | What it does |
|---|---|
| `baselines.py` | Persistence ("next week = this week") baseline in beta space, scored on the same masked entries as the model |
| `sir.py` | **The missing link.** `sir_forward()` runs the SIR incidence equation forwards; `SIRLayer` is the differentiable `nn.Module` version. Also rebuilds the S/I states and evaluates every graph variant in case space |
| `turning_points.py` | Wave-onset detection, and error reported separately on onset weeks vs flat weeks |
| `../training/train_weighted.py` | Retrains with a turning-point-weighted loss. A controlled A/B: `--weight 1.0` reproduces `train.py` exactly |
| `weighted_ablation.py` | Aggregates the weighted/unweighted runs across seeds |
| `risk_ranking.py` | Top-k high-risk district ranking, Precision@k and NDCG@k, plus implied Rt |
| `rolling_origin.py` | Real-time simulation: retrain from scratch for every origin week, predict only that week |
| `make_figures.py` | The four result figures |

## Running it

Everything needs `src` on the path:

    pip install -r requirements.txt

    # 1. train the four graph variants (Person 2's pipeline)
    PYTHONPATH=src python src/training/compare_graphs.py

    # 2. evaluation
    PYTHONPATH=src python src/evaluation/baselines.py
    PYTHONPATH=src python src/evaluation/sir.py
    PYTHONPATH=src python src/evaluation/turning_points.py

    # 3. turning-point weighted loss: 4 graphs x 2 weights x 3 seeds
    for g in geographic gravity learned hybrid; do
      for w in 1.0 5.0; do
        for s in 0 1 2; do
          PYTHONPATH=src python src/training/train_weighted.py \
            --graph $g --weight $w --seed $s
        done
      done
    done
    PYTHONPATH=src python src/evaluation/weighted_ablation.py

    # 4. ranking, rolling-origin, figures
    PYTHONPATH=src python src/evaluation/risk_ranking.py
    PYTHONPATH=src python src/evaluation/rolling_origin.py --graph hybrid
    PYTHONPATH=src python src/evaluation/rolling_origin.py --graph gravity
    PYTHONPATH=src python src/evaluation/make_figures.py

`PYTHONPATH=src` is required — `train.py` imports `models.stgnn` and
`training.data` as top-level packages, which only resolve with `src` on the
path.

## The SIR step

`data.py` builds the training target by inverting the incidence equation:

    beta[t] = new_cases[t] * N / (S[t-1] * I[t-1])

`sir.py` runs it forwards, which is what makes the output usable:

    new_cases[t] = beta[t] * S[t-1] * I[t-1] / N

with `N` = population, `I` = Confirmed − Recovered − Deceased, and
`S` = Population − Confirmed, reconstructed to match `data.py` exactly.

Two consequences: predictions become case counts in units anyone can read,
and the equation bounds them — the model cannot predict more cases than
there are susceptible people, or growth where nobody is infected.

## Results

Test window is the Omicron wave (2021-12-27 to 2022-02-07), 527 districts
with case data, ~354 observed per week.

**1. In beta space, every model loses to persistence.**

| model | MAE | vs persistence |
|---|---|---|
| persistence | 0.4362 | — |
| geographic | 0.5977 | +37.0% |
| gravity | 0.5869 | +34.6% |
| learned | 0.6240 | +43.1% |
| hybrid | 0.5996 | +37.5% |

This reproduces SpatialEpiBench (2026) on our own model, and it is why the
remaining work exists.

**2. Converted to cases, the picture changes.** Hybrid beats persistence
(−1.1%); the others sit within 3% of it. The metric you report changes the
conclusion.

**3. Split by week type, every variant beats persistence where it matters.**

| model | MAE all | MAE onset | MAE flat | onset vs base |
|---|---|---|---|---|
| persistence | 1486.2 | 1627.5 | 1388.9 | — |
| geographic | 1505.2 | 1577.8 | 1455.2 | −3.1% |
| gravity | 1530.1 | 1541.7 | 1522.1 | −5.3% |
| learned | 1513.2 | 1571.4 | 1473.1 | −3.4% |
| hybrid | 1469.5 | 1605.3 | 1375.9 | −1.4% |

Three of four *lose* on the overall average while *winning* at wave onset.
A single aggregate figure ranks these models in the wrong order.

**4. The turning-point weighting helps every graph** (3 seeds each):

| graph | onset MAE, w=1 | onset MAE, w=5 | change |
|---|---|---|---|
| geographic | 1521.5 ± 63.7 | 1439.9 ± 15.5 | −5.4% |
| gravity | 1605.1 ± 90.4 | 1384.5 ± 39.0 | −13.7% |
| hybrid | 1491.7 ± 25.5 | 1409.5 ± 54.8 | −5.5% |
| learned | 1552.9 ± 49.1 | 1467.3 ± 28.6 | −5.5% |

It costs flat-week accuracy (roughly 1400 → 1800 MAE), which is the intended
trade and is reported rather than hidden.

**5. Under rolling-origin evaluation the result holds and strengthens.**
12 origins, each retrained from scratch:

| model | MAE all | MAE onset | onset vs base |
|---|---|---|---|
| persistence | 971.3 | 1409.5 | — |
| hybrid, uniform | 843.8 | 1314.1 | −6.8% |
| hybrid, weighted | 993.7 | 1279.6 | −9.2% |
| gravity, uniform | 838.9 | 1178.7 | −16.4% |
| gravity, weighted | 947.2 | 1132.9 | −19.6% |

Gravity and hybrid with a uniform loss also beat persistence on the overall
average under this protocol (839 and 844 vs 971).

**6. Risk ranking is a mixed result.** On NDCG@10 the models clearly beat
both baselines (≈0.34 vs 0.204 persistence, 0.145 population) — they order
districts by how much they will grow. On Precision@k they lose to
persistence: they are worse at the binary "will it cross the onset
threshold" call.

**7. Implied Rt.** Uniform-loss models produce a mean Rt of 0.94–0.99 over
the Omicron window — below 1, which a growing wave cannot be. The weighted
models give 1.11–1.18. The weighting buys epidemiological plausibility, not
just accuracy.

## Limitations

- **The forecast lags.** `fig4` shows the predicted curve tracking the wave
  but peaking about a week late, as persistence does. Lead time, not
  magnitude, is the next problem.
- **About a third of beta targets are clipped.** The round-trip check in
  `sir.py` recovers only 66% of true cases to within one case, because
  `data.py` clips the target at `beta_max=2.0`. Raising that ceiling should
  help; it caps what the model can learn for the fastest-growing districts.
- **Small sample.** 87 weekly samples, 70 for training. Single-seed results
  swing enough to reverse rankings, which is why every claim above averages
  over 3 seeds or 12 rolling origins.
- **Mild normalisation leakage in rolling-origin.** `prepare_data()`
  normalises features using its own fixed train split, so early origins carry
  a little information from later weeks. Small, and identical across every
  configuration compared.
- **Rt assumes a one-week infectious period** (gamma = 1/week), so the
  absolute values are indicative rather than calibrated.

## Outputs

`outputs/evaluation/` holds the JSON reports and figures (tracked in git).
`outputs/stgnn/` holds checkpoints and raw predictions (gitignored —
regenerate with the commands above).
