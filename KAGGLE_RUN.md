# Running the full experiments on Kaggle

A step-by-step guide for a first-time Kaggle user. You will run **one notebook**, `notebooks/kaggle_run.ipynb`, on a free
Kaggle GPU. It downloads everything, runs all experiments, and gives you **one zip file** to send back.

> **Privacy.** The results contain real LFW faces and deepfake swaps of them. Keep the notebook, its output and every
> dataset you create **Private**. Never post the zip or the qualitative figures publicly.

## 0. Before you start (one time)

1. Create / log in to your account at https://www.kaggle.com.
2. **Verify your phone number**: click your profile picture (top right) -> **Settings** -> **Phone verification**.
   Kaggle only enables **Internet** and **GPU** in notebooks after this. (The notebook needs Internet to download the
   model weights and the LFW dataset.)
3. Make sure the notebook file is on GitHub (`notebooks/kaggle_run.ipynb` on the `main` branch of
   `J4KE-B/face-cloak`). If it has not been pushed yet, use the "upload the file" option in step 1 instead.

## 1. Create the notebook

1. On Kaggle click **+ Create** (left sidebar) -> **New Notebook**.
2. In the notebook menu: **File** -> **Import Notebook**.
   * **From GitHub:** paste `https://github.com/J4KE-B/face-cloak/blob/main/notebooks/kaggle_run.ipynb` (or the raw
     link `https://raw.githubusercontent.com/J4KE-B/face-cloak/main/notebooks/kaggle_run.ipynb`) and import.
   * **Or upload the file:** download `kaggle_run.ipynb` from the repo to your computer, and in the dialog choose the
     upload / "Browse files" option.
3. Delete the empty default cell if Kaggle left one at the top. You should see 9 cells (one text cell, eight code cells).

## 2. Settings (right-hand panel, "Session options" / "Settings")

| Setting | Value |
|---|---|
| Accelerator | **GPU T4 x2** or **GPU P100** (the code uses one GPU either way; a T4 is what the time estimate assumes) |
| Internet | **On** |
| Persistence | **No persistence** (the default; the notebook does its own saving) |
| Environment | "Always use latest environment" is fine |

If Accelerator or Internet is greyed out, your phone is not verified yet (step 0.2). Changing the accelerator restarts
the session, so do it before you run anything.

## 3. Decide what to run (config cell = first code cell)

The first code cell has a `CONFIG` block. One 12-hour Kaggle session is **not enough** for everything (see the
estimate below), so use one of these:

* **Recommended: two sessions, each does half of the people.** Session A: `SHARD = "0/2"`. Session B (a second
  notebook copy, or the same one run afterwards or at the same time): `SHARD = "1/2"`. Each takes roughly 7-9 h.
  Leave everything else as is.
* **Alternative: by experiment.** Session 1: `RUN_CONFIGS = ["main"]`. Session 2:
  `RUN_CONFIGS = ["ablation_eps", "ablation_steps"]`.
* Smaller changes: `REPO_REF` (use a commit SHA of the final code so nothing changes under you),
  `TIME_BUDGET_H = 11.0` (leave it; the notebook stops cleanly at this point and still saves everything).

## 4. Run it

**Recommended: Save & Run All (Commit).** It runs in the background on Kaggle's servers, for up to 12 hours, with
your browser closed and your laptop off.

1. Top right: **Save Version**.
2. Choose **Save & Run All (Commit)**.
3. (If shown under *Advanced settings*) set it to **always save output**, so the zip is kept even if a cell fails.
4. Click **Save**. Close the tab if you like. Check progress via **Your Work** -> your notebook -> **Versions** /
   "View Active Events" (a running version shows a log).

"Run All" (the play-all button in the editor) also works but needs the tab to stay open, and Kaggle stops idle
interactive sessions; use it only for a quick first try, e.g. to check that cells 1-4 pass (about 10-15 minutes).

First-time checks (cells 1-4 should finish in about 10-20 minutes): the GPU line shows `Tesla T4` (or P100) and
`onnxruntime ... CUDAExecutionProvider`; `python -m facecloak.weights --check` shows only `OK`; the manifest shows
`sources=150 targets=30 pairs=450` and `manifest matches the expected fingerprint`. If a cell stops with a message,
read it: they are written to say what to change.

## 5. Expected runtime (estimate, one T4)

This is an **estimate**, not a measurement. Assumptions: 0.06 s per PGD step for the r50 model at 250 px, ensemble
x1.3, EOT x1.2, LPIPS term x1.1, targeted x1.1, detector term x1.8 (a guess); swaps 0.15 s (inswapper) and 0.30 s
(SimSwap); a judge embedding about 0.02 s (the notebook sets `FACECLOAK_ORT_GPU=1`, so the judges and the
calibration run on the GPU; the second number in each cell is the fallback if the GPU provider is unavailable).

| Config | Cloaks (attack x 150 sources) | Swap rows (attacks x 450 x transforms x 2 swappers) | Cloak time | Swap time (judges 0.02 s / CPU judges) | Total |
|---|---|---|---|---|---|
| `main` | 1200 (900 PGD, 100 steps) | 72 000 | 2.4 h | 6.4 h / 9.4 h | **8.8 h / 11.8 h** |
| `ablation_eps` | 750 (all PGD ensemble, 100 steps) | 9 000 | 1.7 h | 0.8 h / 1.2 h | **2.5 h / 2.9 h** |
| `ablation_steps` | 450 (20 / 50 / 100 steps) | 5 400 | 0.6 h | 0.5 h / 0.7 h | **1.1 h / 1.3 h** |
| all three | 2400 | 86 400 | 4.7 h | 7.7 h / 11.3 h | **12.4 h / 16.0 h** |

Plus roughly 0.5-1 h once per session for installs, downloads (weights, LFW), the manifest and the calibration.
So everything on one T4 is about **13-14 h** > 12 h, which is why the work is split in step 3: run **session A
first on your own account**; start session B (the other shard) when A finishes, or in parallel if Kaggle lets
you. With two shards each session needs roughly **7-8 h**. Teammates are only needed if a session fails or your
weekly GPU quota (about 30 h) runs out.

## 6. Get the results

When the committed run finishes (the status turns to "Complete"):

1. Open the notebook -> **Versions** (or **Your Work** -> notebook -> the latest version) -> **Output** tab.
2. Download **`results.zip`** (or `results_shard0of2.zip` / `results_shard1of2.zip` for shards) with the download
   icon next to the file. Size is a few hundred MB (it includes the cloaked images needed for resuming).
3. Also useful: the printed log at the end of the notebook (rows per config, remaining work, any FAILED config).

While it runs, the zip in the Output is refreshed every hour and after every config, so even a crashed session usually
leaves a recent partial zip (the Output of a failed version is still kept if you chose "always save output").

**What to send back:** the zip(s) (share a private Google Drive / Dropbox link, not a public one), the shard/config you
used, and the `repo at <commit>` line from cell 2 and the last cell's summary (a screenshot is fine).

## 7. If the run didn't finish

You get a partial zip (the notebook says so: `stopped at time budget (resumable)` or "missing" in the remaining-work
summary). Nothing is lost: finished work is skipped on the next run.

**Teammates (or you) finish the leftovers:**

1. Download the partial zip. On Kaggle: **+ Create** -> **New Dataset** -> upload the zip -> visibility **Private** ->
   name it e.g. `facecloak-partial` -> **Create**. (Kaggle may unzip it automatically; both forms work.)
2. Open a copy of the notebook (step 1), then right panel -> **Add Input** (or "+ Add Data") -> **Your Datasets** ->
   `facecloak-partial` -> Add. Click the little "copy file path" icon next to the zip to get its path, e.g.
   `/kaggle/input/facecloak-partial/results.zip` (or the folder path if Kaggle unzipped it).
3. In the config cell set `RESUME_FROM = "/kaggle/input/facecloak-partial/results.zip"`.
   To split the leftovers between two people: one sets `SHARD = "0/2"`, the other `SHARD = "1/2"`
   (each with the same `RESUME_FROM`). The notebook reuses the calibration inside the partial zip, so every session
   uses identical thresholds.
4. Run (step 4). Each session produces its own zip.

**Merge everything locally** (on the laptop, in the repo; `calibration.json` is inside the zips):

```bash
unzip -j results.zip calibration.json -d results/            # once, from any of the zips
uv run python scripts/merge_results.py partial.zip finish_a.zip finish_b.zip \
    --out results/raw --cache results/cache/cloaked          # duplicates: first occurrence wins, a warning is printed
uv run python scripts/remaining_work.py --config configs/main.yaml --results results/raw
uv run python scripts/make_figures.py && uv run python scripts/export_web_results.py
```

`remaining_work.py` also prints a recommended shard split if something is still missing.

## 8. Running the live demo backend on Kaggle

That is a different notebook: `notebooks/kaggle_demo_server.ipynb`. Short version: Settings -> GPU + Internet On;
**Add-ons -> Secrets -> add secret `DEMO_KEY`** (a passcode of at least 12 characters); **Run All**; open the link
printed by cell 6; keep the tab open. **Never "Save Version" that notebook with outputs** and never make it public
(the outputs contain the live tunnel URL). Details are in the notebook's first cell and in `server/README.md`.

## Known issues / things to know

* **Judges and calibration use the GPU on Kaggle** (`FACECLOAK_ORT_GPU=1`, set by the notebook). On the laptop
  they stay on the CPU. All reported numbers come from one Kaggle run, so they use one provider consistently.
* **Python version.** `pyproject.toml` says `>=3.11,<3.12`; the notebook installs the package with
  `--ignore-requires-python` and also sets `PYTHONPATH`, so a different Kaggle Python still works.
* **torch version.** Kaggle's preinstalled torch is used (the repo was developed against torch 2.14). Cell 2 prints
  the versions; if SimSwap or the torch graphs fail on an old torch, say so and send the log.
* Free Kaggle accounts have a weekly GPU quota (about 30 h); a full run (both shards) uses roughly 14-16 h of it.
* The qualitative grid (`figs/qualitative*`) is only in the zip, never in `web/`.
