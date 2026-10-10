# face-cloak

**Adversarial Image Cloaking for Protection Against Face-Swap Attacks** — Multimedia Forensics course project.

Face-swap tools (roop / inswapper, SimSwap) turn a source photo into an ArcFace *identity embedding* and paint
that identity onto another face. `face-cloak` adds a small, near-invisible L∞-bounded perturbation (PGD) to a photo
so that the identity embedding — and therefore any swap built from it — no longer matches the real person.

> Status: work in progress. Setup, reproduction and results will be documented here.

## Ethics & licensing

This is a *defensive* research project. Third-party model weights (InsightFace `inswapper_128`, `buffalo_l`,
`antelopev2`, SimSwap) are licensed for **non-commercial research only**; they are downloaded at run time and are
never redistributed in this repository. No face datasets are committed.
