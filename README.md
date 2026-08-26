---
title: Treatment Response Predictor
emoji: 🧬
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Treatment Response Predictor

A composable, **seven-module** machine-learning pipeline with a Streamlit app that
answers one **yes/no** question: will a patient's cancer respond to a given
a treatment (`y = 1` responds, `y = 0` does not), from gene-expression input `x`.

It is built to be usable by **non-experts**: choose your data, click **Build the
model**, and predict — the technical knobs live behind an "Advanced settings"
expander, every control has a plain-language `?` tooltip, and results lead with a
one-line verdict. Everything is **binary by design** ([why](#binary-by-design)).

You pick one implementation per stage and the assembler wires them into a single
trained scikit-learn `Pipeline`.

**This README is in four parts:**
[1 · Use it](#part-1--use-it) · [2 · Run it for others](#part-2--run-it-for-others)
· [3 · Understand it](#part-3--understand-it) · [4 · Project context](#part-4--project-context)

---

# Part 1 — Use it

## Quickstart

```bash
pip install -r requirements.txt
export TML4_APP_PASSWORD="your-password"   # required: the app won't start without one
export TML4_MODEL_KEY="your-model-key"     # required: same, for .tml4model files
streamlit run app.py                       # the GUI
pytest                                     # the test suite
```

Running it where teammates or others can reach it? See
[Part 2](#part-2--run-it-for-others) — there are two secrets to set first.

## Demo run (synthetic data)

No dataset handy? Generate synthetic data for a full end-to-end demo:

1. Start the app: `streamlit run app.py`.
2. In the left **sidebar**, open **Demo data (for testing)** and click
   **Generate demo data** (the 400 × 3000 default is fine).
3. Open **1 · Build a model** — you'll see "Loaded Synthetic: 400 patients ×
   3,000 genes". Leave the defaults and click **Build the model**.
4. Read the **Results** — a plain-language headline plus accuracy / sensitivity /
   specificity, with the confusion matrix and run setup in the expander.
   Optionally **Download PDF report**.
5. *(optional)* In **2 · Save or load**, download the trained model as a locked
   `.tml4model` file (opens only in this app).
6. In **3 · Predict patients**, paste one patient's values or upload a file, predict,
   then open **Why?** for the top genes and **Full prediction report (PDF)**.

The generator plants signal in a few genes so the model scores above chance. It lives
in the sidebar because real users bring their own data.

**Demo images (2-D or 3-D)** are in the same place — switch the demo to *Medical
images*, choose how many slices per patient (1 for flat images, more for a volume
written as a multi-page TIFF), and press generate. The cohort loads straight into step 1
with *Use medical images* selected; one press of **Extract features** measures it. It is
handed over as a zip through the same session key an upload writes, so the demo travels
the real ingestion path — read, measure, features — instead of skipping the code it is
meant to exercise. (The zip is downloadable too, for testing the upload path itself.)
The planted signal is a larger, brighter, rougher lesion in responders, which shape,
intensity and texture features can all see. The default effect size is tuned so the
task is learnable but not trivial (~0.7–0.8 balanced accuracy, not 1.00 — a demo that
separates perfectly can't show the ranking or the reliability flags doing anything).

## The app (two pages, three tabs, one sidebar)

- **Sidebar** — a "Where you are" rail (load data → build a model → predict patients)
  and the demo-data generator. The research-use status sits in the masthead, stated
  once.
- **1 · Build a model** — choose data (your file / the bundled IMvigor210 cohort /
  medical images), name it, then click
  **Build the model**. "Advanced settings" holds the module pickers, the "Test
  rigorously" toggle, CV folds, and the Auto time budget. Results lead with a verdict
  + metric cards; details, confusion matrix and run configuration sit in an expander;
  a PDF report is downloadable. **Compare different setups** evaluates a grid of
  combinations and offers CSV + comparison-PDF downloads.
- **2 · Save or load** — download the trained machine as a locked `.tml4model` file
  (bundled with its configuration + the explainability reference) or upload one to
  predict without rebuilding. Session-only; nothing is cached on the server. See
  [Security](#security--model-files-are-locked-and-authenticated-tml4model).
- **3 · Predict patients** — score one pasted patient or a whole uploaded file; see a
  per-patient results table, **Why?** (top genes per patient), and a downloadable
  cohort **prediction report (PDF)**.
- **Help & FAQs** is a fold at the foot of every page — twelve questions, each folding
  separately, all closed until wanted. Short answers belong where you are working, so
  the sidebar's link points at an anchor *inside* the fold: the browser scrolls there
  **and** opens it in one click.
- **User manual** is its own page at `/manual`, second in the sidebar navigation and
  linked from the page foot. It renders
  [`docs/USER_MANUAL.md`](docs/USER_MANUAL.md), so there is one text — read it in the
  app or on disk. A page rather than a fourth tab (which would make "1 · 2 · 3" read as
  four steps, and tabs have no URL) and rather than a modal (which blocks the app you
  are trying to follow along in). Because it is a real page you can link it, open it in
  a second browser tab, and use the browser's own **Back** button — `st.session_state`
  survives the switch, so the model you just built is still there when you return.

### Auto (pick best), and how long it takes

All three module dropdowns default to **"Auto (pick best)"** — the friendliest choice
when you don't know the methods. Auto evaluates the options on your full data in
parallel across CPU cores, skips combinations that can't run, refuses to pick one whose
result isn't trustworthy (see the honesty guard in [Status](#status)), trains the
winner, and shows the full ranking.

**Auto is thorough, so it takes longer** — a build can't finish before its slowest
single setup does. Three ways to control that:

| Want | Do this |
|---|---|
| A quick result, no ML knowledge | Advanced settings → Prediction method → **Random Forest** |
| A bounded wait, still automatic | Advanced settings → **Time budget for Auto** |
| Fewer cores used (shared node) | `export TML4_MAX_JOBS=8` |

**Time budget** defaults to **No limit**, where every setup is tried so the winner
really is the best available and the result is identical on every run. With a budget
(30 s … 5 min), setups *predicted* to overrun are dropped **before they run** — no
compute is wasted and the app reports exactly what it skipped. The estimate is
calibrated by timing the cheapest setup on *your* data and machine, so it adapts to
100 vs 16,000 genes and to a laptop vs a cluster node.

Skipping is decided up front rather than by killing runs against a stopwatch,
deliberately: a wall-clock cutoff would let machine load decide which setups survive,
so the same data could pick a different winner on a busy node. **The trade-off:** with
a budget the winner is the best of those that *fit*, which might not be the best
overall. Anything skipped stays selectable by hand.

The CV **fold count** also has an "Auto" toggle backed by a data-size heuristic (5 is
standard; 10 with plenty of data) — deliberately *not* a brute-force, since the fold
count shapes the estimate, not the model.

## Data formats (module 1)

- **Upload CSV/TSV** — one row per patient, gene columns first, a final `y` column
  (`0`/`1`). This is the native format.
- **IMvigor210 (ships with the app)** — the one published cohort bundled here: 298
  patients with advanced urothelial cancer on atezolizumab (anti-PD-L1), 68 responders
  (22.8%), 31,286 genes labelled with HGNC symbols, plus tumour mutational burden for
  234 of them. Loaded in one click from *Use a published cohort*. It is bundled because
  it is the real task — immunotherapy response, not tumour-vs-normal — **and** because
  its licence permits it: the `IMvigor210CoreBiologies` package is **CC BY 3.0**, plain
  Attribution, verified in the LICENSE inside the package rather than assumed. See
  [data/IMvigor210_ATTRIBUTION.md](data/IMvigor210_ATTRIBUTION.md) and
  [scripts/bundle_imvigor210.py](scripts/bundle_imvigor210.py), which documents exactly
  what was changed (gene labels Entrez → symbol; no values altered).
- **Upload Tan layout** — genes as rows, patients as columns, text labels in the first
  row; the loader transposes it and maps labels to `0/1`. No Tan file ships with the
  app. These are **tumour-vs-normal** datasets: a pipeline smoke test, *not* a benchmark
  for response prediction — a model can score 90% there and tell you nothing about who
  responds.
- **Synthetic (demo)** — `pipeline/sample_data.generate` plants signal in a few genes;
  available from the sidebar's demo generator.
- **Medical images (module 1b)** — point the app at a folder or zip of scans and it
  measures each one into a fixed panel of named radiomic features (first-order
  statistics, GLCM texture, gradient, difference-of-Gaussians, shape, radial), which
  then flow through the same modules 2–7 as any expression matrix. The measurement
  settings travel *inside* the trained model, so a scan uploaded later in **Predict**
  is measured exactly the way the training cohort was — a panel that differs from the
  training one produces a confident answer that is quietly meaningless.

The **Predict** upload is forgiving: a header row of gene names (e.g. `x_1, x_2, …`)
is detected and skipped, and a trailing 0/1 label column is tolerated.
`convert_leukemia.py` converts the Golub ALL/AML ARFF benchmark into the native format.

### Clinical and genomic columns are worth including (optional)

Every column except the last is a feature, so a column doesn't have to be a gene. If
you have clinical or genomic measurements — **tumour mutational burden, PD-L1 status,
stage, prior therapy** — put them alongside the gene columns and they're used like any
other feature. Nothing to configure, and leaving them out changes nothing.

It is worth doing. On IMvigor210 (298 patients, atezolizumab), adding a single **tumour
mutational burden** column to the full 31,286-gene matrix moved the best result from
**AUC 0.592 → 0.746** (balanced accuracy 0.572 → 0.676) — a bigger gain than any
model or transform choice in this pipeline. That is not surprising: TMB is an
FDA-approved pan-tumour marker of checkpoint response, and expression alone doesn't
capture it.

**One requirement:** every patient needs a value in every column. A clinical field
recorded for only some patients (in that cohort, 64 of 298 had no TMB) will stop the
build — fill the gaps, drop the column, or drop those patients.

*(Tested and not adopted: published response signatures — T-cell-inflamed GEP, IFN-γ,
cytolytic, CD8-effector — scored via Entrez-canonical gene matching. They add nothing
here: `expression + signatures` scored identically to expression alone, and
`expression + TMB + signatures` identically to `expression + TMB`. See `git log`.)*

## Size limits

Spreadsheet uploads are capped at **50 MB**, image cohorts at **500 MB**, and a loaded
matrix at **50 million values**. The two file kinds get different numbers because they
cost different amounts of memory: a CSV expands about 5.6× on its way to float64, so
500 MB of it would peak near **2.8 GB just to load** — an out-of-memory kill for every
visitor sharing the process. A zip of scans is read one image at a time and reduced to
~102 numbers per patient, so it never holds the cohort as pixels.

Streamlit's own `maxUploadSize` is global, so it is set for the larger of the two (500 MB)
and tables are bounded in app code — `check_upload_size` in `pipeline/m1_loader.py`,
raisable per deployment with `TML4_MAX_TABLE_MB`.

Both limits are roughly **5× the largest real cohort here** (IMvigor210, 298 × 31,286),
so ordinary data is unaffected. Over the limit, the app says so with the actual numbers
rather than dying. A server with more RAM can raise the matrix limit:
`export TML4_MAX_CELLS=200000000`.

## Two things to know about the numbers

**Train and predict must use matching data.** A saved model is a fixed pipeline trained
on one specific feature space. Prediction only works when the new patient's `x` has the
**same genes, in the same order** — you cannot train on prostate and predict on a
leukemia patient. The app checks `x`'s length against the model's `n_features_in_` and
refuses on a mismatch, but it **cannot tell** the genes are *different* genes when the
counts happen to match. Keeping train and predict on the same dataset type is on you.

**Headline numbers are out-of-fold (cross-validated)**, not resubstitution: a Random
Forest scores ~100% on its own training data, which says nothing about new patients.
Training accuracy is reported separately for transparency. Sensitivity = recall on
responders; specificity = recall on non-responders.

---

# Part 2 — Run it for others

## The two secrets

Both come from the environment (or `.streamlit/secrets.toml`), and **neither is ever
committed** — `secrets.toml` is gitignored, and a committed
`.streamlit/secrets.toml.example` shows the format.

```bash
export TML4_APP_PASSWORD="the-team-password"   # gates the app
export TML4_MODEL_KEY="the-team-model-key"     # locks/unlocks .tml4model files
```

Guessing is throttled: a session gets five tries before a 60-second wait, and every
wrong answer app-wide adds a delay that grows with recent failures (capped, and never a
global lockout — that would let anyone lock everyone else out). A correct password is
never delayed.

**`TML4_APP_PASSWORD`** gates the whole app. It **fails closed** — with no password set
the app refuses to start rather than accidentally serving itself open. It's *one shared
password*: a doorstop to keep a work-in-progress instance private, **not** real
authentication (no accounts, no roles). A public launch needs proper auth in front of
it. Anyone running a local copy just picks their own; only a *shared* instance's
password needs agreeing on — share it privately, never in the repo.
(Details: [ui/auth.py](ui/auth.py).)

**`TML4_MODEL_KEY`** **fails closed** too, the same as `TML4_APP_PASSWORD` — there is
no built-in default, so with none set the app refuses to start rather than quietly
encrypting model files with a secret that ships in the source. It matters for
**sharing models**: a `.tml4model` file only opens in an app running the **same** key,
so for teammates to exchange saved models, everyone sets the same one. Rotating the
key makes old files stop opening; that's expected.

## Deploying

[DEPLOYMENT.md](DEPLOYMENT.md) is the full checklist, covering three targets:

- **Streamlit Community Cloud** — free, public, zero-ops.
- **Self-managed BU lab server** — systemd or Docker, behind nginx + TLS.
- **Hugging Face Spaces (Docker SDK)** — the current live target. `Dockerfile` and the
  Space metadata at the top of this README configure it, and
  `.github/workflows/sync-to-hf-space.yml` pushes a snapshot to the Space on **every
  push to `merge/as`**.

Set both secrets above in the platform's own secrets store (Space Settings → Variables
and secrets, or Streamlit's Advanced settings) — never in the repo.

## Security — model files are locked and authenticated (`.tml4model`)

A saved model is **not** a portable checkpoint. `pipeline.dumps_machine` pickles the
machine + metadata, then **authenticated-encrypts** the bytes with the app key
(Fernet = AES-128-CBC + HMAC-SHA256) and prepends a magic header. On load,
`loads_machine_bytes` checks the header, then **decrypts-and-verifies before it
unpickles anything**. So:

- **App-locked.** Only an app holding the key can open the file — otherwise it's opaque
  ciphertext, not readable, editable or runnable elsewhere.
- **Tamper-proof.** Any edited byte fails the HMAC check and the file is refused;
  nothing partial is loaded.
- **Closes the pickle RCE hole.** Because decryption authenticates *first*, the code
  only ever unpickles data it encrypted itself. A foreign or malicious upload is
  rejected **before** reaching `pickle.loads`, so it can't execute code on the server.

**Residual risk / operational notes.**
- It is still `pickle` underneath, so the guarantee is "we only unpickle our own
  authenticated bytes." **If the key leaks**, someone could craft a file that passes —
  keep `TML4_MODEL_KEY` secret and rotate it if exposed.
- `TML4_MODEL_KEY` has **no built-in default** — this is deliberate. A key baked into
  the source is only a secret while the source stays private, which this repo can't
  guarantee (a contractor, a code-review tool, a future public mirror all break that
  assumption silently). With no key set, `ui/auth.require_model_key()` stops the app
  before it renders anything, the same way `require_password()` does. See
  [pipeline/assemble.py](pipeline/assemble.py).
- Pin matching library versions for train and deploy so a saved pipeline reloads
  cleanly. Reference: <https://scikit-learn.org/stable/model_persistence.html>

---

# Part 3 — Understand it

## The seven modules

Each module is one file with one job. Stages 1–4 expose an interchangeable **registry**
(a `name -> factory` dict) the GUI reads to offer choices — to add a technique, add one
factory to the registry; nothing else changes.

| Module | File | Registry | What you can toggle |
|--------|------|----------|---------------------|
| 1 Loader        | `pipeline/m1_loader.py`     | `LOADERS`        | Upload CSV/TSV, upload Tan-format, synthetic |
| 1b Imaging      | `pipeline/m1b_imaging_loader.py` | —           | medical images → named radiomic features (`pipeline/imaging/`) |
| 2 Preprocess    | `pipeline/m2_preprocess.py` | `PREPROCESSORS`  | identity, z-score, robust, variance-filter+z-score, log2+z-score |
| 3 Feature map   | `pipeline/m3_feature_map.py`| `FEATURE_MAPS`   | identity, PCA(50), class-conditional PCA, rank transform, select best 500 genes |
| 4 Classifier    | `pipeline/m4_ml.py`         | `MODELS`         | RF, GB, Hist GB, LogReg, SVM(linear/RBF), tree, NB, k-NN, K-TSP, shrunken centroid |
| 5 Evaluate      | `pipeline/m5_evaluate.py`   | —                | cross-validated accuracy / sensitivity / specificity / AUC / confusion, + reliability flags |
| 6 Report        | `pipeline/m6_report.py`     | —                | metrics table, verdict, confusion matrix, prediction + comparison PDFs |
| 7 Explain       | `pipeline/m7_explain.py`    | —                | top genes pushing an individual prediction (predict-time) |

`pipeline/assemble.py` is the **assembler**: `build_machine(preprocess, feature_map,
model)` wires stages 2–4 into one scikit-learn `Pipeline`, so the learned scaling and
feature-map basis travel **with** the classifier. Because the whole machine is one
object it can be kept in the session, downloaded and reloaded as a unit — the Predict
tab just calls `.predict(x)` on it.

**How module 7 stays gene-level:** it occludes one raw gene at a time (resetting it to
the training-set average) and measures how the whole fitted pipeline's `predict_proba`
shifts. Because it runs the *entire* machine, attributions land in the original gene
space even when PCA is active, and it only needs `predict_proba`, so it works for every
classifier.

`Pipeline Flowchart/flowchart.drawio` (diagrams.net) is the editable architecture diagram: every module
box carries its name, key options, and IN/OUT, with each module's OUT matching the next
one's IN.

**Reusing the search from a benchmark harness.** The parallel ordering lives in
`pipeline/scheduling.py` — *not* the GUI — so an external harness gets the same speedup
by importing it:

```python
from pipeline.scheduling import order_combos, max_workers
from joblib import Parallel, delayed

results = Parallel(n_jobs=max_workers(), batch_size=1)(
    delayed(evaluate_one)(c) for c in order_combos(combos)
)
```

`batch_size=1` matters as much as the order: with auto-batching a worker can be handed a
batch containing a slow combo and block while the others idle.

## Binary by design

The pipeline answers one binary question, so the multi-class machinery from the original
`master` branch was intentionally **not** ported: no hierarchical K-TSP tree, no
multi-subtype `LabelManager`, and the Tan loader **refuses files with more than two
classes by default** (`from_tan_file(..., require_binary=True)`), so a multi-class set
raises a clear error instead of silently loading. Supporting multi-class again would
mean `require_binary=False` **plus** a multi-class-aware classifier and evaluator.

## Project layout

```
app.py               Streamlit entry point — password gate, page config + tabs -> render()
conftest.py          puts the repo root on sys.path so tests import the package
pipeline/            importable core (no Streamlit dependency)
  __init__.py        re-exports build_machine / train / dumps_machine / order_combos / ...
  assemble.py        assembler: wires modules 2–4 into one machine; train/save/load; compat checks
  m1_loader.py       module 1 — loaders (CSV/TSV, Tan-format, synthetic) + header auto-skip
  m2_preprocess.py   module 2 — preprocessors
  m3_feature_map.py  module 3 — feature maps (PCA adapts its component count)
  m4_ml.py           module 4 — classifiers
  m5_evaluate.py     module 5 — cross-validated evaluation, suggest_folds, reliability flags
  m6_report.py       module 6 — tables / verdict / confusion / prediction + comparison PDFs
  m7_explain.py      module 7 — per-prediction explainability (top genes)
  m1b_imaging_loader.py  module 1b — medical images -> named radiomic features
  imaging/           image readers, preprocessing, and the radiomic feature panel
  estimators.py      custom techniques (K-TSP, shrunken centroid, class-conditional PCA,
                       AdaptivePCA, AdaptiveHistGradientBoosting)
  scheduling.py      parallel-search ordering + time budget — shared with any harness
  sample_data.py     synthetic benchmark generator
ui/                  Streamlit pages and tabs — one render() each
  nav.py             the two pages + why the manual is a page, not a tab or a modal
  workflow.py        the workflow page: the three numbered tabs
  auth.py            password gate (fails closed if no password is set)
  theme.py           design system — type scale, tokens, masthead, page/section
                     headers, verdict card, graduated rule, notices, progress rail
  sidebar.py         progress rail (painted after the tabs) + demo-data generator
  train.py           build / evaluate / report / compare + Auto + time budget
  model.py           save / load a trained model (.tml4model)
  predict.py         multi-patient predict + Why? (top genes) + cohort report
  state.py           shared session-state helper (store_dataset + the size guard
                       every data source passes through)
  imaging_page.py    the medical-image source picker, shown inside the Build tab
  help.py            the FAQ fold at every page foot + the manual page (from docs/)
docs/
  USER_MANUAL.md     the user manual — one text, read on disk or inside the app
.streamlit/
  config.toml        disables the file watcher (hot-reload broke model pickling)
  secrets.toml.example  template for the app password — copy to secrets.toml (gitignored)
tests/               pytest suite
Dockerfile           image for the HF Space and the BU server
DEPLOYMENT.md        deployment checklist (3 targets)
Pipeline Flowchart/
  flowchart.drawio   editable architecture diagram (diagrams.net)
  Image.jpg          exported flowchart image (embedded in Part 5)
  Flowchart.pdf      exported flowchart, print version
convert_leukemia.py  Golub ALL/AML ARFF -> app-ready CSV
requirements.txt
```

---

# Part 4 — Project context

## Where each teammate's technique went

The four independent branches (`master`, `as`, `zz`, `mvp-pipeline`) were each a full
attempt; `merge/as` (via `merge/simplified` / `mvp-pipeline`) folds the useful parts
into one pipeline. Checked against the branches themselves:

| Capability | Came from (branch) | Now in `merge/as` |
|------------|--------------------|-------------------|
| K-TSP / Top-Scoring Pairs | `master` (`Ktsp_model.py`) | `estimators.KTSPClassifier` + `m4_ml` |
| Nearest shrunken centroid | `master` (`multi_model_trainer.py`) | `estimators.NearestShrunkenCentroid` + `m4_ml` |
| Class-conditional PCA | `master` (`SubspaceModule.py`) | `estimators.ClassConditionalPCA` + `m3` |
| Model-registry pattern + sklearn zoo | `master` (`models_registry.py`) + `as` (`algorithms/`) | `m4_ml` `MODELS` |
| Tan-format loader | `as` (`tan_data.py`) | `m1_loader.from_tan_file` |
| Feature-map registry (PCA / rank / variance) | `as` (`feature_map/`) | `m3_feature_map` |
| Preprocessing (z-score / robust / variance filter) | `as` (`feature_map/standard_scaler`, `top_k_variance`) | `m2_preprocess` |
| Synthetic data generator | `as` (`data_generator.py`) / `mvp-pipeline` (`sample_data.py`) | `sample_data` |
| **Per-prediction explainability (top genes)** | **`as` (`app.py`, SHAP-based)** | **`m7_explain`** (re-done as occlusion) |
| Predict column check + trailing-marker tolerance | `as` (`app.py`) | Predict-tab guard (accepts N or N+1) |
| Six-module GUI structure | `zz` (`module1..6`) + `mvp-pipeline` (`m1..m6`) | `m1..m6` + `ui/` |
| PDF report | `zz` (`module6_evaluation.py`, fpdf) | `m6_report` |
| Save / load trained model | `zz` (`model_store.py`, joblib) + `as` | `ui/model.py` + `dumps/loads_machine` |
| Cross-validated evaluation | `mvp-pipeline` (`m5_evaluate.py`) | `m5_evaluate` |
| Docker / HF Space deployment | `merge/as` (evandugas) | `Dockerfile`, `.github/workflows/` |

Note that **explainability (module 7) originated as Angelina's SHAP work on the `as`
branch** — here re-expressed as occlusion so it stays model-agnostic and reports in gene
space even through the feature map.

## Status

**What the Tan benchmarks are for.** The eight Tan datasets (colon, prostate ×3,
leukemia, lung, DLBCL, GCM) classify **tumour vs normal tissue** — not treatment
response. They are a *smoke test*: they prove the pipeline runs correctly across a
range of sizes and shapes, from 33 to 280 patients and 2,000 to 16,000 genes. They are
**not** the performance claim, which is why the perfect scores on the small ones are a
warning rather than an achievement. Response-prediction performance is measured on the
immunotherapy cohorts (IMvigor210, GSE78220).

**One clinical column beats 26,128 genes (IMvigor210, 2026-08-14).** Three arms on the
same 234 patients — TMB alone, expression alone, and both — 561 evaluations. Best
trustworthy balanced accuracy: **TMB alone 0.662** (AUC 0.720), genes+TMB 0.655,
expression alone 0.608. **Zero of 288 gene-based setups beat the single TMB column**,
and the comparison favours the genes if anything: they were given 275 attempts each
against TMB's 11. With 61 responders the SE is ~0.036, so no single pairwise gap here is
decisive — the "0 of 288" pattern is what carries it. The practical reading: on this
cohort, clinical covariates and a calibrated threshold matter more than the choice of
model, which is why the app asks for those columns at upload.

Verified on the 8 real Tan benchmarks (colon, prostate ×3, leukemia, lung, DLBCL, GCM).
`git log` has the blow-by-blow; the short version:

- **Auto ranks on the balanced score, not raw accuracy.** The balanced score is the
  two recalls averaged — it is 0.5 for any constant guess *at any prevalence*, so a
  setup can't win by answering the same way for nearly everyone. Raw accuracy can:
  on IMvigor210 (23% responders) it ranked a setup finding **16 of 68** responders
  above one finding **41 of 68**. All four numbers — accuracy, sensitivity,
  specificity, balanced — are still shown, each with a one-line explanation of what it
  counts. Across 10 datasets the change moved the recommended setup on 4, and on three
  of those the sensitivity was identical (a tie-break between equally good models).
- **Compare, ranked your way.** The comparison table can be ranked by balanced score,
  accuracy, catching responders, ruling out non-responders, or AUC — each with a banner
  saying what that choice optimises *and* what it costs, because which mistake matters
  more is a clinical judgement, not a statistical one. The balanced pick is always named
  with its justification, including what ranking by accuracy would have chosen instead.
- **Reliability.** Module 5 flags any result that isn't real learning: a model that
  predicts one class for everyone, one that finds almost none of a group, one no better
  than chance once both groups count equally, or one whose solver never converged.
  Flagged setups show in red and are excluded from Auto's pick, so a fake
  class-balance accuracy can't be reported as a result. "Better than chance" is judged
  on **balanced** accuracy, not raw accuracy against the majority baseline — on an
  imbalanced cohort a genuinely useful model often scores *below* that baseline
  (IMvigor210: K-TSP at .644 accuracy vs a .772 baseline, but sens .60 / spec .66).
- **Degenerate combos: 153 of 160 closed.** Profiling found four classifiers silently
  collapsing; all traced to scikit-learn defaults that don't hold for wide, small-n
  genomics data, and all fixed — SVM (RBF) `gamma` (126 → 0, now a top performer),
  Naive Bayes `var_smoothing` (11 → 0), Hist GB `min_samples_leaf` (16 → 0), plus
  adaptive PCA components so small cohorts run at all. Remaining: class-conditional-PCA
  → linear SVM (5) and k-NN (2), both flagged by the guard.
- **Fair cross-validation.** Custom estimators declared their mixins in an order that
  lost the classifier tag under scikit-learn ≥1.6, so K-TSP and Nearest Shrunken
  Centroid were scored with *unstratified* CV while sklearn's classifiers got
  stratified. Fixed — but **their pre-fix benchmark numbers aren't comparable with the
  rest of the table** (both were underrated; DLBCL K-TSP .688→.779, NSC .662→.792).
- **Bounded runtime.** No single fit can hang the app (one pathological combo once ran
  3.4 h; now < 2 s), and the Auto search dispatches slowest-first with an optional time
  budget.
- **Performance.** ~70–84% parallel efficiency at 8 cores; CPU-bound, no GPU needed. A
  single ordinary multi-core machine runs the full grid in minutes.

## Open decisions

Judgment calls left for the team rather than made unilaterally:

- **Re-profile first.** Four fixes since the last benchmark change reported numbers, and
  the K-TSP / NSC CV fix makes their old numbers incomparable. The decisions below
  should be made on a fresh run, not the current table.
- **A single default method.** Recurring top performers are **Logistic Regression,
  linear SVM, K-TSP** and now **SVM (RBF)**, on rank-transformed or identity features.
  Auto tries everything by default; if you want one fast default, those are the
  evidence-backed picks. Note the benchmark table needs restating under the balanced
  ranking (below), which changed the best setup on 4 of 10 datasets.
- **Gradient Boosting on full gene counts.** Classic GB and Hist GB both time out above
  a few hundred patients, and Hist GB did not deliver the speedup it was added for
  (~2× total compute, no lower floor) — though its collapses are now fixed, so this is a
  **compute** decision, not a correctness one. Options: drop full-gene-count GB/HistGB
  combos, restrict them to reduced feature maps, or keep them and rely on the time
  budget.
- **How to present perfect scores.** Leukemia / Lung / Prostate3 hit 1.000 — on 30–180
  patients against 12k+ genes that's an overfitting / easy-benchmark warning, not a
  triumph. Present as "the pipeline runs correctly," not "the model is 100% accurate."


# Part 5 — Pipeline Flowchart

![alt text](<Pipeline Flowchart/Image.jpg>)