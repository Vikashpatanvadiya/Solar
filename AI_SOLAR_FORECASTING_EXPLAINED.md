# Is This Project AI? And How Does It Predict Solar Power?

*Last verified against the code and results on 21 October 2026. Every number below was
produced by `solar-prediction/src/run.py` and `src/regions.py` on this machine — nothing
here is illustrative or rounded up.*

---

## 1. The short answer

**Yes — this is a machine-learning system, and machine learning is a branch of AI.**

But it is worth being precise, because "AI" is used loosely. This project is **not** a
chatbot or a large language model. It is a **supervised regression** system: it is shown
thousands of real examples of *"this was the weather, this is how much solar power India
generated"*, it learns the relationship between the two, and it is then asked to predict
days it has never seen.

That learning-from-examples step is what makes it AI rather than a formula. Nobody wrote
a rule saying "if sunlight is 6.2 and humidity is 40%, output 520 MU". The model worked
that relationship out on its own from 640 days of history.

### What it would need to NOT be AI
If the project just averaged last year's generation, or applied a fixed efficiency
equation, it would be statistics or physics — not AI. It does neither. It fits
**six different learning algorithms** and lets the evidence pick the winner.

---

## 2. The six models in this project

All six live in `solar-prediction/src/models/`. Each one learns the same thing in a
different way, which is the point — we can show that the result is not an artefact of
one lucky algorithm.

| # | Model | File | How it learns |
|---|-------|------|----------------|
| 1 | **Linear Regression** | `linear.py` | Fits one straight-line equation through all the data. The honest baseline — any real model must beat it. |
| 2 | **Decision Tree** | `decision_tree.py` | Builds a flowchart of yes/no questions ("was GHI above 5.8?") that splits days into groups with similar output. |
| 3 | **SVM (SVR)** | `svm_model.py` | Support Vector Regression with a Gaussian **RBF kernel** — bends the data into a higher-dimensional space where a straight surface *can* fit a curved relationship. |
| 4 | **GPR** | `gpr_model.py` | Gaussian Process Regression with a **Rational Quadratic kernel** + white-noise term. A Bayesian method: instead of one answer it models a whole probability distribution over possible answers. **This is the production model.** |
| 5 | **ANN** | `ann.py` | A feed-forward neural network, written from scratch in NumPy: 2 hidden layers of 10 **sigmoid** neurons, trained by **back-propagation with momentum**. |
| 6 | **ANN-PSO** | `ann_pso.py` | The same neural network, but its weights are found by **Particle Swarm Optimization** (`pyswarms`) instead of back-propagation — a swarm of candidate weight-sets "flies" through weight-space toward the lowest error. |

> **Note on the neural network.** It uses **sigmoid** activations and **gradient descent
> with momentum**, implemented by hand in NumPy — *not* ReLU and *not* the Adam optimiser.
> This follows the reference paper (Kumar et al.) rather than modern defaults, and it
> avoids a TensorFlow dependency that will not install on this machine's Python.

---

## 3. How a prediction is actually made, step by step

```
NASA POWER satellites                 Grid-India PSP reports
(sunlight, temp, wind, humidity)      (daily solar generation, MU)
          │                                       │
          ▼                                       ▼
 capacity-weighted average               divide by installed capacity
 over 12 solar-hub states                   → capacity factor
          │                                       │
          ▼                                       │
 NOCT physics → panel temperature                 │
          │                                       │
          └──────────────► FEATURES ◄─────────────┘
                              │
                     train on 2024–2025
                              │
                              ▼
                     predict 2026 capacity factor
                              │
                     × that month's installed capacity
                              ▼
                   predicted generation in MU
```

### Step 1 — Get the real weather (NASA POWER)
Daily satellite observations from NASA's Langley Research Center, free and without an
API key, for 12 locations that represent where India's panels actually are:

| Hub | Place | Region | Fleet weight |
|-----|-------|--------|--------------|
| Rajasthan | Bhadla / Jodhpur | NR | 22% |
| Gujarat | Charanka, Patan | WR | 14% |
| Karnataka | Pavagada | SR | 12% |
| Andhra Pradesh | Kurnool | SR | 9% |
| Tamil Nadu | Kamuthi | SR | 8% |
| Telangana | Hyderabad belt | SR | 7% |
| Maharashtra | Shirdi / Ahmednagar | WR | 6% |
| Madhya Pradesh | Rewa | WR | 6% |
| Rajasthan West | Bikaner | NR | 6% |
| Uttar Pradesh | Kanpur / Jhansi | NR | 4% |
| Bihar–Jharkhand | Gaya / Hazaribagh | ER | 2% |
| Assam | Guwahati | NER | 0.5% |

India is 3.2 million km². A plain geographic average would let empty desert and empty
ocean-facing coast count as much as Bhadla. So each hub is weighted by its share of
national capacity:

$$\text{National weather}_t = \sum_{i=1}^{12} w_i \times \text{Weather}_{i,t}$$

### Step 2 — Add panel temperature from physics (NOCT)
Panels lose roughly 0.4% efficiency per °C above 25 °C, and a panel in desert sun runs
far hotter than the air around it. So we compute module temperature rather than
guessing it, using the standard NOCT model:

$$T_m = T_a + \frac{G}{800}\,(\text{NOCT} - 20), \qquad \text{NOCT} = 45^\circ\text{C}$$

Daily GHI (kWh/m²/day) is converted to an effective daytime irradiance $G$ (W/m²) by
spreading the day's energy over an assumed 8 sun-hours. **This is an explicit
approximation** — the reference paper used hourly data; we only have daily.

### Step 3 — Solve the fleet-growth problem (capacity factor)
This is the single most important design decision in the project.

India's installed renewable capacity went from **144,752 MW (Apr 2024)** to
**207,093 MW (Dec 2025)** — **+43% in 21 months**. If the model predicted raw MU from
weather, it would conclude that 2026 sunshine is far more powerful than 2024 sunshine,
when in truth there are simply more panels.

So the model never predicts MU directly. It predicts **capacity factor**:

$$CF = \frac{\text{Daily generation (MU)}}{\text{Installed capacity (MW)}}$$

and converts back at the end:

$$\widehat{\text{Generation}} = \widehat{CF} \times \text{Installed capacity that month}$$

For 2024–2025 the **actual** monthly capacity is used. 2026 is beyond the published
table, so capacity is extrapolated using the **compound** growth rate of the last 12
months (≈2.0%/month nationally). Compound rather than straight-line, because capacity
additions build on a bigger fleet each month — switching to compound improved the
national forecast from 5.46% to **5.04% error** and cut the systematic bias from
−2.9% to −1.5%.

### Step 4 — Seasonal features
Daily data has no "time of day", but it does have a position in the year. Day-of-year is
encoded as two continuous coordinates so that 31 December sits next to 1 January:

$$\text{doy}_{\sin} = \sin\!\left(\frac{2\pi \cdot \text{DOY}}{365.25}\right), \qquad
  \text{doy}_{\cos} = \cos\!\left(\frac{2\pi \cdot \text{DOY}}{365.25}\right)$$

### Step 5 — Scale, train, predict
All inputs and the target are MinMax-scaled to [0, 1] **using training-set statistics
only**, so the 2026 test year never influences the scaling. The split is strictly
chronological — never shuffled — because shuffling a time series lets the model peek at
the future.

- **Train:** 640 days, 1 Apr 2024 → 31 Dec 2025
- **Test:** 274 days, 1 Jan 2026 → 15 Oct 2026 *(never seen during training)*

---

## 4. The seven inputs the model gets

| Input | Unit | Correlation with generation |
|-------|------|------------------------------|
| **GHI (sunlight)** | kWh/m²/day | **+0.57** — strongest driver |
| **Relative humidity** | % | **−0.51** — haze and monsoon cloud cut output |
| **Module temperature** | °C | +0.31 |
| **Wind speed** | m/s | −0.25 |
| **Ambient temperature** | °C | +0.04 — almost no direct signal on its own |
| doy_sin, doy_cos | — | seasonal position |

That is the whole input list. No generation history is fed in, so the model cannot
cheat by copying yesterday's answer.

---

## 5. Results — what it actually achieves

**Test: 274 days of 2026 the models never saw. Accuracy = 100 − MAPE.**

### With module temperature

| Model | Accuracy | MAPE | RMSE (MU) | MAE (MU) | R² |
|-------|---------:|-----:|----------:|---------:|----:|
| **GPR** | **95.0%** | **5.04%** | **37.7** | **29.7** | **0.710** |
| SVM | 94.9% | 5.11% | 38.6 | 30.3 | 0.696 |
| ANN | 94.5% | 5.47% | 40.0 | 32.4 | 0.673 |
| Linear Regression | 94.1% | 5.86% | 43.2 | 35.0 | 0.618 |
| Decision Tree | 93.7% | 6.29% | 46.6 | 37.6 | 0.557 |
| ANN-PSO | 93.6% | 6.42% | 47.7 | 38.4 | 0.534 |

### Without module temperature

| Model | Accuracy | MAPE | RMSE (MU) | R² |
|-------|---------:|-----:|----------:|----:|
| GPR | 94.9% | 5.08% | 37.9 | 0.706 |
| SVM | 94.9% | 5.09% | 38.5 | 0.697 |
| ANN-PSO | 94.9% | 5.12% | 37.4 | 0.714 |
| ANN | 94.3% | 5.66% | 42.1 | 0.638 |
| Linear Regression | 94.1% | 5.86% | 43.2 | 0.618 |
| Decision Tree | 93.8% | 6.24% | 46.1 | 0.565 |

**What this comparison says, honestly:** module temperature helps the plain ANN
(94.3% → 94.5%) but makes almost no difference to GPR or SVM, and it *hurts* ANN-PSO.
On daily national data, module temperature is largely a re-expression of sunlight and
air temperature, which the model already has. We report this rather than claiming the
physics term is a breakthrough.

### What the error metrics mean

- **MAE** — on an average day the model is off by about **30 MU** out of ~585 MU.
- **RMSE** — like MAE but punishes big misses harder; 37.7 MU.
- **MAPE** — average error as a percentage: 5.04%. Accuracy = 100 − 5.04 = **95.0%**.
- **R²** — the fraction of day-to-day variation explained: 0.710, i.e. 71%.

---

## 6. Proving it is real, not a lucky test set

A single good score proves little. The model is checked **two independent ways**
(`src/validate_holdout.py`):

| Test | Period | Days | Accuracy | RMSE | R² |
|------|--------|-----:|---------:|-----:|----:|
| Hidden past data | 1 Jul – 31 Dec 2025 | 184 | **94.3%** | 29.8 | 0.638 |
| Future year | 1 Jan – 15 Oct 2026 | 274 | **95.0%** | 37.7 | 0.710 |

The two agree to within **0.7 percentage points**. A model that had memorised its
training data, or got lucky on one window, would not score the same on two separate
hidden periods.

---

## 7. Region-by-region forecasts

Each of the five Grid-India regions gets **its own model**, trained on its own hub
weather and its own installed capacity (`src/regions.py`). The national figure is never
just split up proportionally.

| Region | Accuracy | RMSE (MU) | R² | 2026 actual | 2026 predicted |
|--------|---------:|----------:|----:|------------:|---------------:|
| Southern (SR) | **94.5%** | 10.3 | 0.649 | 43,988 MU | 42,205 MU |
| Northern (NR) | 91.5% | 24.0 | 0.471 | 63,046 MU | 66,966 MU |
| Eastern (ER) | 85.9% | 0.6 | 0.559 | 1,058 MU | 1,051 MU |
| Western (WR) | 85.8% | 33.6 | 0.185 | 51,869 MU | 44,868 MU |
| North-Eastern (NER) | 75.7% | 0.6 | 0.061 | 324 MU | 364 MU |

**Known weakness, stated plainly:** Western Region is the weakest (85.8%, under-predicting
by 13%). The capacity table gives *total* renewable capacity (`res_mnre_MW`), which in
Gujarat and Maharashtra includes a great deal of **wind**. WR's solar grew faster than
its total-renewable figure suggests, so the capacity yardstick is too small there. The
honest fix is solar-only region-wise capacity data, which Grid-India's PSP reports do not
publish. We have not invented it.

---

## 8. Why the 2026 data used to be incomplete — and how it was fixed

**The original problem:** 2024–2025 came from Grid-India's *monthly* PSP PDFs (one file
per month, every day inside). 2026 has no monthly PDFs yet, so it had to come from
*daily* Excel files — and only 124 days had been downloaded, in scattered blocks
(Feb 11–28, all of March, Apr 1–25, Aug, Sep 1–22). Jan, May, Jun and Jul were missing
entirely, which is why only March ever showed a complete comparison.

**Why guessing the URLs did not work:** the daily files have random suffixes —
`11.02.26_NLDC_PSP_193.xls`, `12.02.26_NLDC_PSP_792.xls`. There is no pattern to
construct.

**The fix:** the Grid-India site is a JavaScript app backed by a public JSON API. Reading
its own bundle gave the endpoint:

```
POST https://webapi.grid-india.in/api/v1/file
{"_source":"GRDW","_type":"DAILY_PSP_REPORT","_fileDate":"2026-27","_month":"00"}
```

That returns the true file path for every published report. 158 missing days were added
to the manifest and downloaded.

**Result: 2026 coverage went from 124 days to 281 days (January → October).**

| Month | Days now available |
|-------|-------------------|
| Jan 2026 | 27 / 31 |
| Feb 2026 | 27 / 28 |
| Mar 2026 | 31 / 31 |
| Apr 2026 | 29 / 30 |
| May 2026 | 30 / 31 |
| Jun 2026 | 29 / 30 |
| Jul 2026 | 29 / 31 |
| Aug 2026 | 30 / 31 |
| Sep 2026 | 30 / 30 |
| Oct 2026 | 19 / 31 (month in progress) |

> ### ⚠️ A correction to the earlier version of this document
> A previous automated edit "fixed" the missing months by **scaling partial months up to
> full months**:
> `monthly_actual = (sum of recorded days / days recorded) × days in month`.
>
> That is not a fix — it manufactures actuals that Grid-India never published, and then
> measures the model's accuracy against them. The accuracy figures it produced
> (e.g. September 2026 "actual" 18,546 MU from 22 days) were not real measurements.
> **That behaviour is now off by default** in `scripts/import_actuals_from_daily.py` — it
> has to be asked for explicitly with `--scale-partial`, and it prints a warning that the
> result is not a published measurement. The dashboard compares only days that genuinely
> exist and shows gaps as gaps; partial months in the monthly view are labelled as totals
> of the days available, not estimates of the whole month.

---

## 9. Forecasting 2027, and checking it against an outside opinion

2026 could be *scored*, because the weather actually happened and Grid-India published
what was generated. 2027 cannot. So the 2027 view in the dashboard is labelled a
**scenario**, and it is built from two stated assumptions.

### Assumption 1 — a typical year of weather (climatology)
For every day of the year, take the median of that same day across 2024, 2025 and 2026,
smoothed with a ±3-day window (a single calendar date has only 3 observations — far too
few for a stable median; the window gives ~21). A typical year has no heatwave, no failed
monsoon, no unusual cloud. **Real 2027 will differ, and the 95.0% daily accuracy measured
on 2026 does not carry over** — in 2026 the model was handed the weather that really
occurred.

What the model still contributes is the **shape**: how output responds to irradiance,
heat, wind and humidity, and how that plays out across the seasons.

### Assumption 2 — a projected fleet, shown as a range
Installed capacity is published only to Dec 2025, so 2027 is projected two ways:

| Fleet path | Dec 2027 capacity | GPR annual total |
|---|---:|---:|
| Low growth (straight-line) | 300,054 MW | **243,796 MU** |
| Recent growth (compound) | 336,444 MW | **263,588 MU** |

An 8% spread — **the fleet assumption, not the model, is the biggest unknown in 2027.**
The dashboard lets you switch between the two rather than hiding the choice.

### All six models agree
Trained on all 914 days of history (2024-04-01 → 2026-10-15) and run on the same typical
year, the six algorithms land within **2.1%** of each other:

| Model | 2027 total (recent-growth fleet) |
|---|---:|
| SVM | 264,210 MU |
| GPR | 263,588 MU |
| ANN | 262,449 MU |
| Linear Regression | 262,249 MU |
| Decision Tree | 262,168 MU |
| ANN-PSO | 258,881 MU |

Six very different algorithms reaching nearly the same answer is a good sign. The
dashboard draws this as a green consensus band — where it widens, confidence is lower.

### The independent benchmarks
A model compared only with itself proves nothing, so two baselines are computed from
completely different ingredients (`solar-prediction/src/external_benchmarks.py`):

**1. NREL PVWatts physics.** Pure engineering, no parameter learned from Indian data:

$$E_{ac} = \underbrace{G \times k_{tilt}}_{\text{plane of array}} \times \big[1 + \gamma (T_{cell} - 25)\big] \times (1 - L_{sys}) \times \eta_{inv}$$

with PVWatts v8 defaults — system losses 14.08%, inverter efficiency 96%, module power
coefficient −0.0047/°C — plus two stated assumptions of ours: a 1.08× tilt gain, and that
solar is 56% of the reported MNRE renewable fleet (the capacity table counts wind and
biomass too). If `NREL_API_KEY` is set the live PVWatts v8 API is queried per hub and
cached; otherwise the same published formula is evaluated locally. Either way it falls
back cleanly if the service is unreachable.

**2. Persistence.** The "do nothing clever" forecast: carry 2026's observed capacity
factor forward onto the projected fleet, ignoring weather entirely.

### Is the benchmark itself any good? — we checked
The PVWatts baseline was run on **real 2026 weather** and scored against **real 2026
generation**:

| | Accuracy | Bias | RMSE |
|---|---:|---:|---:|
| NREL PVWatts physics | 92.6% | −5.4% | 51.6 MU |
| **Our GPR model** | **95.0%** | −1.5% | **37.7 MU** |

This is the most useful single result in the project. A physics model with no training at
all gets within 7.4% of India's daily solar output — which says the problem is largely
physical. **Our model beats it by 2.4 points on the same 274 days**, and that margin is
what the machine learning is actually worth: it has learned the part of the behaviour the
textbook formula misses (soiling, curtailment, grid effects, the real tilt mix).

### 2027 side by side

| Source | 2027 total | vs our GPR |
|---|---:|---:|
| Persistence (2026 capacity factor) | 279,624 MU | −5.7% |
| **Our GPR model** | **263,588 MU** | — |
| NREL PVWatts physics | 250,798 MU | +5.1% |

Our forecast sits between the two, which is where it should be: above pure physics
(because real fleets out-perform the conservative default loss assumptions) and below
naive persistence (because 2026 was a relatively good solar year).

> **No CEA / MNRE published target is bundled with this repo.** `config.CEA_ANNUAL_TARGET_MU`
> is left as `None` and the dashboard simply omits that row. Paste an official published
> figure there and it appears automatically — we did not want to print a government number
> we could not cite.

### API

```
GET  /api/forecast/2027      scenario: 6 models x 2 fleet paths, regions, consensus band
POST /api/forecast/2027      recompute it
GET  /api/comparison/2027    our model vs every benchmark, month by month, with divergence
```

Years outside `config.FORECAST_YEARS` return **400** with an explanation rather than a
silently wrong answer.

---

## 10. What this model cannot do

Stated up front, because a model you can't criticise isn't a model you can trust.

1. **It cannot predict a specific day in 2027.** It needs *that day's actual weather* as
   input, and weather forecasts are only useful ~10 days out. The 2027 view (§9) is a
   climatology scenario — an annual and monthly shape built on typical weather — not a
   day-level forecast, and it must never be quoted with the 95.0% figure.
2. **R² of 0.71, not 0.99.** The reference paper reports R² = 0.9992, but it predicts a
   *single rooftop array at the same instant* from on-site sensors. We predict *a whole
   country's next-year daily total* from satellite weather. These are different problems
   and the paper's number is not reachable here. We have not tuned the numbers to match it.
3. **Western Region is weak** (see §7) — a data limitation, not a modelling trick.
4. **Capacity beyond Dec 2025 is extrapolated**, so late-2026 predictions carry extra
   uncertainty on top of the model's own error.
5. **Only 21 months of training data.** Less than two full seasonal cycles.

---

## 11. Running it yourself

```bash
cd solar-prediction
.venv/bin/python src/fetch_weather.py     # NASA POWER (cached)
.venv/bin/python src/preprocess.py        # build the dataset
.venv/bin/python src/run.py               # train all 6 models + regions
.venv/bin/python src/validate_holdout.py  # the independent check
.venv/bin/python src/external_benchmarks.py  # score PVWatts physics on real 2026
.venv/bin/python src/forecast_future.py      # build the 2027 scenario
.venv/bin/python -m uvicorn src.server:app --port 8010
```

Then open **http://127.0.0.1:8010** — a single page with the forecast, the model picker,
the reliability check, the 2027 outlook, the weather, the map, and the raw data. There is no separate
"AI lab" page; everything is on the one site.

---

### Sources
- **Generation:** Grid-India (POSOCO) PSP reports — <https://grid-india.in/en/reports/daily-psp-report>
- **Weather:** NASA POWER, Langley Research Center — <https://power.larc.nasa.gov>
- **Method:** Kumar et al. (2026), solar PV output prediction with MRMR feature selection,
  NOCT module temperature, and six regression models including ANN-PSO.
