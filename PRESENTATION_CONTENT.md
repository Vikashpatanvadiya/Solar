# Presentation Content — Solar Power Generation Forecasting

Slide-by-slide content for you to build the deck.
For each slide: **Title**, the **text to put on the slide**, the **visual** to add, and
**speaker notes** (what to say out loud — do not put this on the slide).

Suggested length: **16 slides, ~12 minutes.**
Every number below is real and verified — nothing here is placeholder.

---

## Slide 1 — Title

**On the slide**
- Predicting India's Daily Solar Power Generation Using Machine Learning
- Your name · Renewable Energy Technology
- 95.0% accuracy on 274 days of unseen 2026 data

**Visual:** a screenshot of the dashboard home view.

**Say:** "I built a system that predicts how much solar electricity India generates each
day, using only that day's weather. It's 95% accurate on days it had never seen."

---

## Slide 2 — The problem

**On the slide**
- India's renewable fleet grew **43% in 21 months** — 144,752 MW → 207,093 MW
- Solar is intermittent: output depends entirely on weather
- Too little solar → coal/gas must start at short notice (slow, expensive)
- Too much solar → clean power gets curtailed and wasted
- Grid-India publishes what **was** generated. Nobody publishes what **will be**.

**Visual:** a simple bar showing capacity growth Apr 2024 → Dec 2025.

**Say:** "Grid operators have to balance supply and demand every single minute. If they
don't know how much solar is coming, they either burn fuel they didn't need, or throw
away clean electricity."

---

## Slide 3 — The question

**On the slide**
> Given a day's weather, how much solar electricity will India generate that day —
> and can a machine learn that relationship well enough to be trusted?

**Visual:** keep this slide nearly empty. One sentence, large.

**Say:** "That's the whole project in one sentence. Note the second half — 'well enough
to be trusted'. Proving the number is as important as getting it."

---

## Slide 4 — The data

**On the slide**

| Source | What it gives | Size |
|---|---|---|
| Grid-India PSP reports | Daily solar generation (MU) | 2024–2026 |
| NASA POWER satellites | Sunlight, temp, wind, humidity | 12 locations |
| Monthly capacity table | Installed MW, nationally and per region | Apr 2024 – Dec 2025 |

- Final modelling dataset: **914 usable days**
- Train: 640 days (Apr 2024 – Dec 2025) · Test: 274 days (Jan – Oct 2026)
- 2024–25 from monthly PDFs; 2026 from **293 daily Excel files**

**Visual:** the data-flow diagram (Section 3.1 of the Notion doc).

**Say:** "All of it is real published data. No simulated or synthetic values anywhere in
the training or testing."

---

## Slide 5 — Why 12 locations, not one

**On the slide**
- India is 3.2 million km² — a plain average would weigh empty desert the same as Bhadla
- Weather is **capacity-weighted** across 12 real solar hubs
- Rajasthan (Bhadla) 22% · Gujarat (Charanka) 14% · Karnataka (Pavagada) 12% · Andhra
  (Kurnool) 9% · Tamil Nadu (Kamuthi) 8% · and 7 more

Formula: National weather = Σ wᵢ × Weatherᵢ

**Visual:** map screenshot from the dashboard with the 12 hub markers.

**Say:** "The weather that matters is the weather where the panels actually are."

---

## Slide 6 — The physics we built in (NOCT)

**On the slide**
- Panels lose ~0.4% efficiency per °C above 25 °C
- A panel in desert sun is far hotter than the air around it
- So panel temperature is **computed**, not guessed:

Tₘ = Tₐ + (G / 800) × (NOCT − 20),  NOCT = 45 °C

**Visual:** a small diagram — sun → panel → heat → efficiency drop.

**Say:** "This is the one place where physics is injected directly into the machine
learning. It comes from the reference paper."

---

## Slide 7 — The key insight: capacity factor ⭐

**On the slide**
- **Problem:** the fleet grew 43%. The same sunny day produces far more power in 2026
  than in 2024 — just because there are more panels.
- A model predicting raw MU would learn "2026 sunshine is stronger". That's wrong.
- **Solution:** never predict MU. Predict **capacity factor**:

CF = Daily generation (MU) ÷ Installed capacity (MW)

Then convert back: Predicted MU = CF × capacity that month

- Result: R² went from **−10** to **positive**.

**Visual:** two panels — "raw MU: trend upward, model confused" vs "capacity factor: flat
and stable".

**Say:** "This is the single most important decision in the project. Before this fix the
model was worse than just guessing the average. This is what made it work."

> This is your strongest slide. Spend time here.

---

## Slide 8 — The six models

**On the slide**

| Model | How it learns |
|---|---|
| Linear Regression | One straight-line equation — the honest baseline |
| Decision Tree | A flowchart of yes/no questions |
| SVM (RBF kernel) | Bends data into higher dimensions to fit curves |
| **GPR** | Bayesian — models a probability distribution of answers |
| ANN | Neural net, 2 hidden layers of 10 sigmoid neurons, back-propagation |
| ANN-PSO | Same net, weights found by a particle swarm instead |

- The neural network is **written from scratch in NumPy**, not TensorFlow

**Visual:** the ANN layer diagram.

**Say:** "Six models, not one. If six completely different algorithms agree, the result
isn't an accident of one lucky method."

---

## Slide 9 — How it was trained fairly

**On the slide**
- Split is **strictly chronological** — never shuffled
- Train 2024–2025 → Test 2026
- Scaling statistics come from training rows **only**
- The model is never shown generation history — it cannot copy yesterday

**Visual:** a timeline bar: TRAIN (640 days) → TEST (274 days).

**Say:** "If you shuffle a time series, the model trains on days from inside the test
period and effectively sees the future. The accuracy looks amazing and means nothing.
Every split here is in time order."

---

## Slide 10 — Results

**On the slide**

| Model | Accuracy | RMSE (MU) | R² |
|---|---|---|---|
| **GPR** | **95.0%** | **37.7** | **0.710** |
| SVM | 94.9% | 38.6 | 0.696 |
| ANN | 94.5% | 40.0 | 0.673 |
| Linear Regression | 94.1% | 43.2 | 0.618 |
| Decision Tree | 93.7% | 46.6 | 0.557 |
| ANN-PSO | 93.6% | 47.7 | 0.534 |

- Average miss: **30 MU** out of ~585 MU a day
- 274 days total: actual 160,284 MU vs predicted 157,810 MU — a 1.5% gap

**Visual:** horizontal accuracy bars (the dashboard leaderboard screenshot).

**Say:** "Accuracy here means 100 minus the average percentage error."

---

## Slide 11 — Proving it's real, not luck

**On the slide**
- One good score proves nothing. So it was tested **twice, independently**:

| Test | Period | Days | Accuracy |
|---|---|---|---|
| Hidden past data | Jul – Dec 2025 | 184 | **94.3%** |
| Future year | Jan – Oct 2026 | 274 | **95.0%** |

- The two agree within **0.7 percentage points**

**Visual:** two large side-by-side figures: 94.3% and 95.0%.

**Say:** "A model that had just memorised its training data would not score the same on
two separate hidden periods. This is what makes the number believable."

---

## Slide 12 — Beating an industry-standard baseline ⭐

**On the slide**
- NREL **PVWatts** is the standard engineering model — pure physics, nothing learned
- We ran it on the **same real 2026 days**:

| Method | Accuracy | Bias | RMSE |
|---|---|---|---|
| NREL PVWatts physics | 92.6% | −5.4% | 51.6 MU |
| **Our GPR model** | **95.0%** | −1.5% | **37.7 MU** |

- **Our model beats it by 2.4 percentage points**

**Visual:** two bars, 92.6% vs 95.0%, ours highlighted.

**Say:** "This is the result I'm most proud of. A textbook physics formula with zero
training already gets within 7.4% — which tells you the problem is mostly physical. Our
model beats it by 2.4 points, and that gap is exactly what the machine learning is
worth: it has learned the things the formula can't model — dust on panels, curtailment,
grid effects, the real mix of panel tilts."

> If the examiner asks only one question, it will probably be about this slide.

---

## Slide 13 — Region by region

**On the slide**

| Region | Accuracy | 2026 actual | 2026 predicted |
|---|---|---|---|
| Southern | 94.5% | 43,988 MU | 42,205 MU |
| Northern | 91.5% | 63,046 MU | 66,966 MU |
| Eastern | 85.9% | 1,058 MU | 1,051 MU |
| Western | 85.8% | 51,869 MU | 44,868 MU |
| North-Eastern | 75.7% | 324 MU | 364 MU |

- Each region has **its own model**, its own weather, its own capacity
- Western is weakest: the capacity table counts wind too, which distorts Gujarat

**Visual:** the interactive map screenshot with region cards.

**Say:** "The national number is never just split up proportionally — each region is
modelled separately. And I'm showing you the weak one too, with the reason."

---

## Slide 14 — Looking ahead to 2027

**On the slide**
- 2027 weather hasn't happened, so this is a **scenario**, not a measured forecast
- Weather = a typical year (median of 2024–2026, day by day)
- Capacity = projected, shown as a **range**, not one number

| Source | 2027 total |
|---|---|
| Persistence baseline | 279,624 MU |
| **Our GPR model** | **263,588 MU** |
| NREL PVWatts | 250,798 MU |

- All six models agree within **2.1%**

**Visual:** the 2027 chart with the green consensus band.

**Say:** "The 95% accuracy does NOT apply here, and the dashboard says so in red. For
2026 the model was given the weather that really happened. For 2027 it's given an
average. What it still gives us is the shape and the scale."

---

## Slide 15 — Honest limitations

**On the slide**
- Cannot predict a **specific day** in 2027 — weather forecasts only work ~10 days out
- R² is 0.71, not the paper's 0.9992 — the paper predicts one rooftop at one instant
  from on-site sensors; we predict a whole country's daily total for a future year
- Western Region is weak — a data limitation (no solar-only regional capacity published)
- Capacity after Dec 2025 is projected
- Only 21 months of training data — under two full seasonal cycles

**Visual:** none needed. Plain text.

**Say:** "I'm putting the weaknesses on a slide on purpose. A model you can't criticise
isn't a model you can trust. I didn't tune any number to match the paper."

> Examiners reward this. It also pre-empts the hardest questions.

---

## Slide 16 — Conclusion

**On the slide**
- **95.0%** accurate on 274 unseen days
- Confirmed by a second independent test at **94.3%**
- **Beats** the industry-standard physics model by 2.4 points
- One dashboard: forecast · 6 models · reliability · weather · map · 2027
- Every number traceable to a published source; every assumption written down

**Visual:** the full dashboard screenshot.

**Say:** "Thank you — happy to take questions."

---

# Likely questions, and short answers

**"Is this really AI?"**
Yes — supervised machine learning. Nobody wrote the rule connecting weather to output;
the model learned it from 640 days of examples. It's not a chatbot, but it is AI.

**"Why is R² only 0.71 when the paper got 0.9992?"**
Different problem. The paper predicts one rooftop array at the same instant from on-site
sensors. We predict a whole country's daily total for a future year from satellite
weather. I did not tune our numbers to match theirs.

**"How do I know it isn't overfitting?"**
Three ways: the split is chronological so it never sees the future; scaling uses training
data only; and it scores 94.3% and 95.0% on two separate hidden periods.

**"Why GPR and not the neural network?"**
GPR won on the test data — 95.0% vs 94.5%. I report what won, not what sounds most
impressive.

**"Why does ANN-PSO do worse than in the paper?"**
Particle Swarm Optimization struggles in high-dimensional weight-space, so the network
had to stay small. It came last for us. I'm reporting that rather than hiding it.

**"Where does the 2026 data come from?"**
Grid-India's daily PSP Excel files. The filenames have random suffixes so URLs can't be
guessed — I found their public JSON API and pulled 158 missing days, taking 2026 from
124 days to 281.

**"Could this be used in production?"**
For planning and day-ahead support, with a real weather forecast as input, yes. For
minute-by-minute grid balancing you'd need intra-day data, which isn't published.

---

# Screenshots to capture

Run the dashboard first:

```bash
cd /Users/bansi/Developer/Solar/solar-prediction && .venv/bin/python -m uvicorn src.server:app --port 8010
```

Then open **http://127.0.0.1:8010** and capture:

1. Top of page — KPI tiles + monthly actual-vs-predicted chart → slides 1, 16
2. "How the six models compare" leaderboard → slide 10
3. "Is the model reliable?" two tiles → slide 11
4. "2027 outlook" chart with the green band → slide 14
5. The map with region cards → slides 5, 13
6. "The weather the model reads" tiles → optional extra

Also available as ready-made images in `solar-prediction/results/plots/`:
`1_actual_vs_predicted_line.png`, `2_actual_vs_predicted_scatter.png`,
`3_residuals.png`, `4_monthly_box.png`, `5_model_comparison.png`
