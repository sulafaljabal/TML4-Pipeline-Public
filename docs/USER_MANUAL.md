# Treatment Response Predictor — user manual

A step-by-step guide to using the app. No machine-learning background assumed: if you
can read a spreadsheet of patients, you can use this.

> **Research demo · not for clinical use.** This tool supports research and
> exploration. It is not a diagnostic device, it has not been validated for clinical
> decision-making, and a prediction here should never be the sole basis for a treatment
> decision.

---

## Contents

1. [What the tool does](#1-what-the-tool-does)
2. [Signing in](#2-signing-in)
3. [Preparing your data](#3-preparing-your-data)
4. [Step 1 — Build a model](#4-step-1--build-a-model)
5. [Reading the results](#5-reading-the-results)
6. [Comparing setups](#6-comparing-setups)
7. [Step 2 — Save or load a model](#7-step-2--save-or-load-a-model)
8. [Step 3 — Predict patients](#8-step-3--predict-patients)
9. [Working from medical images](#9-working-from-medical-images)
10. [Trying it without data](#10-trying-it-without-data)
11. [When something goes wrong](#11-when-something-goes-wrong)
12. [What the numbers mean](#12-what-the-numbers-mean)
13. [Limits — what this tool cannot tell you](#13-limits--what-this-tool-cannot-tell-you)

---

## 1. What the tool does

You give it patients whose outcome is already known — their gene-expression profiles
and, for each one, whether they responded to a treatment. It finds the pattern that
separates responders from non-responders, tells you honestly how well that pattern
holds up on patients it was never shown, and then applies it to new patients.

The answer is always binary: **1 = responds**, **0 = does not respond**. There is no
partial or multi-class outcome. That is a deliberate simplification, not a limit of
your data.

Three steps, used in order, one tab each. Each uses what the previous one produced:

| Step | Tab | You end up with |
|---|---|---|
| 1 | **Build a model** | a trained model + an honest score for it |
| 2 | **Save or load** | that model as a file you can reopen later *(optional)* |
| 3 | **Predict patients** | a call, a confidence, and an explanation per patient |

The sidebar's **Where you are** rail shows which of the three are done.

---

## 2. Signing in

The app asks for one shared password before it renders anything. This is a doorstop
that keeps stray visitors off a development URL — not real authentication. There are no
individual accounts, and nothing here should be treated as an access-controlled
deployment. If you don't have the password, ask whoever shared the link.

Five wrong answers in a session start a short wait, and repeated failures across the
whole app slow every further wrong answer down. Neither ever slows down someone who
knows the password.

---

## 3. Preparing your data

**Shape.** One row per patient. One column per gene. A **final column of 0s and 1s**
saying whether that patient responded. The first row holds the column names (there's a
checkbox for files that don't have them).

```
gene_A , gene_B , gene_C , ... , response
  7.21 ,   4.02 ,   6.55 , ... ,        1
  6.88 ,   5.14 ,   6.01 , ... ,        0
```

**Format and size.** CSV, TSV or TXT, up to **50 MB** and 50 million cells. Medical
images arrive as a zip instead, up to **500 MB** — bounded separately because a zip is
read one image at a time rather than loaded whole. Anything larger is refused with a
message rather than left to exhaust the server's memory.

**Which label means "responded"** is asked after loading — *Which group responded to
treatment?* — so a file coded the other way round, or with `R`/`NR`, still works.

**Clinical and genomic columns help, and are optional.** If you have measurements
beyond expression — tumour mutational burden, PD-L1, stage, prior therapy — put them in
as extra columns before the response column. They're treated as features like any gene.
This is worth doing, and the measurement behind that claim is unusually stark. On the
bundled IMvigor210 cohort we compared three arms on the *same* 234 patients: tumour
mutational burden alone, 26,128 genes alone, and both. TMB alone scored **0.662**
balanced accuracy; the genes alone reached **0.608**; and **not one of 288 gene-based
setups beat the single TMB column**. One clinical measurement carried more than the
entire expression panel. If you have such measurements, include them — they are likely
to matter more than which method you pick. Every patient needs a value in every column.

**Practical minimums.** Nothing enforces a patient count, but the honest floor for a
result you can believe is roughly 40–50 patients with at least ~15 in the smaller
group. Below that, cross-validation estimates swing widely between runs, and the app
will often flag the result as unreliable — correctly.

To make that concrete: what limits you is the size of the **smaller** group, because
sensitivity is measured only on those patients. With 14 responders among 38 patients,
the 95% interval on the balanced score is about **±0.16** — so 0.60 and 0.70 are the
same result, and nothing below roughly 0.65 can be told apart from chance. With 61
responders among 234, that interval tightens to about ±0.07. Adding patients to the
larger group barely helps; adding them to the smaller one helps a lot.

---

## 4. Step 1 — Build a model

**Choose your data** — *Where should the data come from?*

- **Upload my file** — your own spreadsheet, as above.
- **Use a published cohort** — **IMvigor210**, which ships with the app: 298 patients
  with advanced urothelial cancer treated with atezolizumab (anti-PD-L1), 68 of them
  responders (22.8%). One click loads it. This is the real thing the tool is for —
  actual immunotherapy outcomes — so it is the honest place to see what the app can and
  cannot do.

  A checkbox adds **tumour mutational burden** as a feature. Worth trying: on this
  cohort that single clinical column moves AUC further than any modelling choice in the
  app. It costs patients — only 234 of the 298 have a TMB value, and the other 64 are
  *dropped rather than filled in*, because imputing a fifth of a column would invent the
  signal you are trying to measure.

  Data: IMvigor210CoreBiologies (Mariathasan et al., *Nature* 554:544–548, 2018), used
  under [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/).

  Under the same option you can also upload a file in the **Tan benchmark layout**
  (genes in rows, patients in columns, labels on the top row); the app transposes it and
  reads the labels for you. No Tan file ships with the app. Those are
  **tumour-vs-normal** problems — useful for checking the pipeline runs end to end, but
  not treatment-response data, so a good score there says nothing about predicting who
  responds.
- **Use medical images** — see [section 9](#9-working-from-medical-images).

Name the dataset if you like (it labels the reports), then click **Build the model**.

### Advanced settings (optional)

Everything here has a sensible default; you can build a model without opening it.

- **Three method dropdowns**, one per pipeline stage. All default to **Auto (pick
  best)**:
  - **Normalize** — None (identity), Standardize (z-score), Robust standardize
    (median/IQR), Variance filter + standardize, Log2 + standardize.
  - **Transform (optional)** — None (identity), PCA (50 components), Class-conditional
    PCA, Rank transform (TSP-style), Select best 500 genes.
  - **Prediction method** — Random Forest, Gradient Boosting, Hist Gradient Boosting,
    Logistic Regression, SVM (linear), SVM (RBF), Decision Tree, Naive Bayes, k-NN,
    K-TSP (Top-Scoring Pairs), Nearest Shrunken Centroid.
- **Test rigorously (recommended)** — cross-validation. Leave it on. With it off you
  get a fast number that is over-optimistic, because the model is being graded on
  patients it studied.
- **Number of folds (data splits)** — how many equal parts the patients are split into.
  Each part is held out in turn and predicted by a model trained on the rest. **5 is
  the standard choice**; use 10 with plenty of patients, fewer when a group is small.
  More folds means each test model trains on more data but runs slower and gives a
  noisier estimate.
- **Time budget for Auto** — no limit (most accurate), or roughly 30 s / 1 / 2 / 5
  minutes. With a budget set, Auto spends it on the combinations most likely to pay off
  and reports what it skipped.

### What Auto actually does

It evaluates every combination of preprocessing × feature map × model on *your* data,
in parallel across CPU cores, discards combinations that can't run, refuses to pick one
whose result isn't trustworthy, trains the winner on everything, and shows you the full
ranking.

Two honest caveats:

- **It is slower**, necessarily — a build can't finish before its slowest single setup
  does. For a quick answer, pick one specific method instead of Auto, or set a time
  budget.
- **The winner's score runs a little optimistic.** The number reported is the same
  estimate that was used to choose it out of everything compared, so it is best read as
  an upper bound — more so on a small dataset, and more so when many setups were
  compared. The results page repeats this next to the number itself.

---

## 5. Reading the results

The results lead with a **verdict** and metric cards; the detail sits under **See
detailed metrics, confusion matrix, and run setup**. A PDF report is downloadable.

- **Accuracy** — of all patients, how many were called correctly.
- **Sensitivity** (*catches responders*) — of the patients who did respond, how many
  the model found.
- **Specificity** (*rules out non-responders*) — of those who didn't respond, how many
  it correctly cleared.
- **Balanced score** — sensitivity and specificity averaged. This is what Auto ranks
  by, because a model cannot win it just by answering the same way for nearly everyone
  — which is the main way a model looks good and is useless.
- **Confusion matrix** — the four counts behind all of the above: correct and incorrect
  calls, split by which direction the mistake went.

**The borderline band.** Every percentage in the app is read against the same 0–100
scale with the midpoint marked. Anything within 10 points of 50% is shaded
**borderline**: statistically close to a coin flip and not a confident call either way.

**The reliability flag.** The app refuses to present a result as trustworthy when the
evidence says it isn't. A setup is flagged when it:

- didn't converge (the solver hit its iteration limit and gave up),
- predicted only one class (not learning — just answering the same way every time),
- found almost none of one group, or
- didn't beat random chance.

A flag is information, not a bug. Try a different setup, more patients, or different
preprocessing.

---

## 6. Comparing setups

**Compare different setups** evaluates a grid of combinations and shows them in one
table, downloadable as CSV and as a comparison PDF.

**Rank these setups by** decides what "best" means, because which mistake costs more is
a clinical judgement, not a statistical one. A banner under the control spells out what
the current ranking optimises for:

| Ranking | Use it when |
|---|---|
| **Balanced score** *(recommended)* | the default; both groups count equally |
| **Accuracy** | you care about overall hit rate — but read it against the class split: if 8 in 10 patients don't respond, answering "no" every time already scores 80% |
| **Catches responders** | missing a responder is the worse mistake (a patient is denied a treatment that would have worked) |
| **Rules out non-responders** | a false alarm is the worse mistake (a patient gets a toxic, ineffective treatment) |
| **ROC AUC** | you want ranking quality rather than a yes/no call — how well the model orders patients by risk, independent of where the cut-off sits. Useful when you plan to choose your own threshold later |

Whichever ranking you choose, the **balanced pick is still named** alongside it, so a
ranking that suits one question doesn't quietly become the default.

---

## 7. Step 2 — Save or load a model

A model you build lives only in your browser session. Closing the tab loses it. Nothing
you upload or train is written to the server's disk.

- **Download this model (.tml4model)** bundles the trained model, its configuration and
  the reference data behind the per-patient explanations into one file.
- **Uploading** one restores it, so you can predict without rebuilding.

The file is encrypted and app-locked: it opens only in this app, and a file that has
been modified — or produced anywhere else — is refused rather than silently loaded. It
can't be edited or read as plain data outside the tool. It is not a portable checkpoint
for other software, by design.

---

## 8. Step 3 — Predict patients

Two ways in: **paste one patient's gene values** separated by commas, or **upload a
patient file** — same columns as the training data, in the same order, without the
response column. A trailing response column is tolerated and ignored, so the same file
you trained on can be re-scored.

You get:

- **A call per patient** — responds or does not — with the probability behind it, read
  against the same 0–100 scale, borderline band and all.
- **The cohort split** — how many of the uploaded patients fall each way.
- **Why? — top genes behind a prediction**, per patient: which genes pushed the call and
  in which direction. Pick the patient by row number and how many genes to show.
- **A prediction report (PDF)** for the whole cohort, and the predictions as CSV.

An explanation is a description of *this model's* reasoning. It is not evidence that a
gene is biologically causal.

---

## 9. Working from medical images

Choose **Use medical images** in step 1. The app measures the images for you and turns
each patient into a row of numbers, which then travels the ordinary pipeline.

- **Upload a `.zip`** of the cohort. A zip, not loose files, because browsers discard
  folder names when you select many files at once — and the folders are what group
  images by patient.
- **Layout**: `<outcome>/<patient>/<image files>`, so the outcome comes from the folder
  name exactly as a real cohort is usually filed. Alternatively include a
  `manifest.csv` at the top of the zip and that is used instead.
- **Readable formats**: `.bmp`, `.jpeg`, `.jpg`, `.png`, `.tif`, `.tiff`, `.webp`.
  Multi-page TIFFs are read as a stack of slices, which is how a volume arrives.
- **Which part of the image to measure** — *Automatic (find the object)* or *Whole
  image*.
- **What to measure** — the feature families, each adding a block of numbers:
  brightness in the region (20), brightness of the whole image (8), texture (32), edges
  and sharpness (12), detail at several scales (8), shape of the region (16),
  core-to-rim profile (6). Two more are **off by default because they can encode which
  machine took the picture** — image size / raw range (6), and colour channels (10), for
  stained slides. A model that learns the scanner instead of the disease scores well and
  means nothing, so those are opt-in.
- Region previews sit under *What exactly was measured? (region previews)*, a
  plain-language description of every feature under *What each feature means*, and the
  measured table downloads as CSV.

A patient with many slices still becomes **one row** — the slice measurements are
aggregated. Depth therefore makes the measurements steadier; it does not currently add
a new *kind* of information.

---

## 10. Trying it without data

Open **Demo data (for testing)** in the sidebar.

- **Gene expression** — choose how many training patients, how many genes, and how many
  held-out sample patients. It generates a synthetic dataset with a planted signal,
  loads the training part straight into step 1, and offers the held-out patients as a
  download you can feed into step 3.
- **Medical images** — choose patients, image size, and slices per patient (1 for flat
  2-D images, more for a 3-D volume written as a multi-page TIFF), then press
  **Generate demo images**. The cohort is loaded straight into step 1 with *Use medical
  images* already selected; press **Extract features** to measure it, exactly as a real
  cohort needs. Responders get a larger, brighter, rougher lesion. The same zip is also
  downloadable if you want to test the upload path itself — it is foldered by outcome
  like a real cohort.

The demo is for trying the tool end to end, not for drawing conclusions. The signal is
planted; a real dataset is the only thing that tells you anything about biology.

---

## 11. When something goes wrong

| What you see | What it means | What to do |
|---|---|---|
| *"No password is set, so this app won't start"* | The deployment has no `app_password` / `TML4_APP_PASSWORD` configured | Whoever runs the app must set one — see `.streamlit/secrets.toml.example` |
| *"No model-encryption key is set"* | Same, for `TML4_MODEL_KEY` — the key that locks `.tml4model` files | As above; models saved under a different key won't open |
| File refused as too large | A spreadsheet over 50 MB, an image zip over 500 MB, or over 50 million cells | Cut columns (e.g. keep the most variable genes) or split the cohort |
| *Column mismatch* on prediction | The patient file's columns don't match what the model was trained on | Same genes, same order, no response column (one trailing column is tolerated) |
| Result flagged unreliable | The honesty guard caught one of the four failures in [section 5](#5-reading-the-results) | Try another setup, or more patients — don't report the number |
| A build takes a long time | Auto is trying every combination | Set a **Time budget for Auto**, or pick one method instead of Auto |
| A saved model won't open | Wrong encryption key, a modified file, or one made elsewhere | Use the key it was saved with; otherwise rebuild |

---

## 12. What the numbers mean

**Cross-validation** splits patients into *k* folds, holds each one out in turn, trains
on the rest and predicts the held-out part. Every patient gets predicted exactly once,
by a model that never saw them. That is why it estimates performance on *new* patients,
where scoring the training data does not.

**Why balanced score is the default.** Suppose 20% of your patients respond. A model
that answers "does not respond" for everyone is 80% accurate and completely useless: it
finds no responders at all. Its balanced score is 50% — exactly chance, which is the
honest reading. Accuracy alone hides this; the balanced score cannot.

**Sensitivity and specificity trade off.** Pushing one up generally pushes the other
down. Which you want depends on which mistake costs more, and that is a decision about
patients, not about statistics — which is why the app asks rather than choosing for you.

**AUC** summarises performance across every possible threshold instead of a single
cut-off. 0.5 is chance; 1.0 is perfect.

---

## 13. Limits — what this tool cannot tell you

- **It is not clinically validated.** Research and exploration only.
- **A model is only as good as its cohort, and this is not a hypothetical.** Trained on
  one population it may not transfer to another — different centre, different assay,
  different treatment. We measured it: a model trained on IMvigor210 (bladder cancer,
  anti-PD-L1) and tested on GSE78220 (melanoma, anti-PD-1) transferred **not at all**,
  in either direction, across 1,100 combinations — including the scale-free methods that
  should have been most robust to the platform change. Treat a score as belonging to the
  cohort it was measured on until you have checked otherwise.
- **Correlation, not mechanism.** The genes named under *Why?* are what this model
  leaned on. They are not proof of biology.
- **Small cohorts flatter models.** With few patients, a good-looking score is often
  luck. The reliability flag catches the worst cases, not all of them.
- **The Auto winner's score is an upper bound**, for the reason in
  [section 4](#4-step-1--build-a-model).
- **Nothing is stored.** Data and models exist only for the life of your browser
  session. Save the `.tml4model` file if you need the model tomorrow.
