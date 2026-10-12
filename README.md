# face-cloak

**Adversarial Image Cloaking for Protection Against Face-Swap Attacks** — Multimedia Forensics course project
(IIIT Kottayam).

Face-swap tools (roop / inswapper, SimSwap) turn a source photo into an ArcFace *identity embedding* and paint
that identity onto another face. `face-cloak` adds a small, near-invisible L∞-bounded perturbation (PGD) to a photo
so that the identity embedding — and therefore any swap built from it — no longer matches the real person.

## 🌐 Live demo website

**https://face-cloak.vercel.app** — upload a photo, cloak it, and watch a face swap work on the original but fail on
the cloaked copy.

👉 **[How to use the website](docs/WEBSITE_GUIDE.md)** — starting the backend (Kaggle GPU or laptop), connecting,
running the demo, reading the result, privacy and troubleshooting.

## Documentation

| Guide | For |
|---|---|
| [docs/REPORT.md](docs/REPORT.md) | **Full project report** (method, setup, results, discussion, limitations) |
| [docs/PRESENTATION_BRIEF.md](docs/PRESENTATION_BRIEF.md) | Everything needed to build the slides: outline, speaker notes, key numbers, diagrams, demo runbook, viva Q&A |
| [docs/WEBSITE_GUIDE.md](docs/WEBSITE_GUIDE.md) | Presenting / trying the live demo |
| [KAGGLE_RUN.md](KAGGLE_RUN.md) | Running the full experiments on a free Kaggle GPU |
| [server/README.md](server/README.md) | Demo backend (FastAPI): configuration, API, security |
| [web/README.md](web/README.md) | Frontend (Next.js) development and deployment |
| [report/](report/) and [slides/](slides/) | IEEE report and Beamer slides (LaTeX) |

## How it works (short)

1. Detect the face and its 5 landmarks.
2. Compute the ArcFace identity embedding.
3. Optimise an L∞-bounded perturbation (ε = 8/255 by default) with PGD so the embedding moves away from the real
   identity (ensemble, targeted decoy, EOT and LPIPS variants available).
4. Attack the cloaked photo with a real face-swap model (inswapper, SimSwap).
5. Judge the swap with a **held-out** face recogniser (glintr100) the cloak never saw; protection = similarity below
   a threshold calibrated on LFW at 1 % false-accept rate.

## Repository layout

```
facecloak/   core package: models, differentiable alignment, PGD cloak, swappers, metrics, experiment runner
scripts/     data prep, calibration, experiments, figures/tables, web export
configs/     experiment configs (smoke, main, ablations)
server/      live-demo backend (FastAPI) + laptop tunnel script
web/         live-demo website (Next.js, deployed on Vercel)
notebooks/   Kaggle notebooks: full experiments, GPU demo backend
tests/       pytest suite (light tests run anywhere; parity/real-model tests need weights)
report/      IEEE report (LaTeX)        slides/   Beamer slides
```

## Quick start (development)

```bash
uv sync                                    # Python 3.11 environment
uv run python -m facecloak.weights --fetch # download + sha256-check model weights
uv run pytest -q -m "not slow"             # light tests
```

## Ethics & licensing

This is a *defensive* research project. Third-party model weights (InsightFace `inswapper_128`, `buffalo_l`,
`antelopev2`, SimSwap) are licensed for **non-commercial research only**; they are downloaded at run time and are
never redistributed in this repository. No face datasets are committed. The live demo only swaps onto fixed
public-domain portraits, watermarks every synthetic image, and keeps uploads in memory for at most 10 minutes.
