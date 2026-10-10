"""API contract of the live-demo backend (Pydantic v2). The frontend's TypeScript types mirror this file."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

JobState = Literal["queued", "running", "done", "error", "expired"]
Stage = Literal["detecting", "cloaking", "swapping", "judging"]
Mode = Literal["fast", "robust"]

CONSENT_NOTICE = (
    "Only upload photos of yourself or people who consented. "
    "Images are processed in memory and deleted within 10 minutes."
)


class Health(BaseModel):
    status: Literal["ok", "loading"]
    version: str
    device: str = Field(description="'cuda' or 'cpu'")
    models_loaded: bool
    queue_length: int = Field(description="jobs queued + running")
    default_steps: int
    max_eps: int
    tau: float = Field(description="glintr100 identity threshold at FAR=1%")
    max_steps: int = Field(description="largest accepted PGD step count on this server's device")
    consent_notice: str = Field(description="text the frontend must show next to the upload control")


class Target(BaseModel):
    id: str
    name: str
    thumbnail: str = Field(description="data:image/jpeg;base64,... (<=256 px)")
    source: str = Field(description="URL of the original image")
    license: str


class JobCreated(BaseModel):
    job_id: str
    queue_position: int = Field(description="0 = running now / next, 1 = one job ahead, ...")


class Progress(BaseModel):
    stage: Stage
    step: int = Field(description="PGD step (estimated from elapsed time while cloaking), else 0")
    total: int = Field(description="total PGD steps")


class Scores(BaseModel):
    id_sim_original: float | None = Field(description="cosine(judge(swap of original), reference)")
    id_sim_cloaked: float | None = Field(
        description="null if the attacker could not even detect the cloaked face"
    )
    tau: float
    judge: Literal["glintr100"] = "glintr100"
    reference: Literal["uploaded", "self"] = Field(
        description="'uploaded' = separate reference photo, 'self' = original upload"
    )


class Quality(BaseModel):
    psnr: float
    ssim: float
    lpips: float | None
    linf_255: int


class Timings(BaseModel):
    cloak_s: float
    total_s: float


class Params(BaseModel):
    eps: int
    steps: int
    mode: Mode


class JobResult(BaseModel):
    original: str = Field(description="PNG data URL, long side <= 1024")
    cloaked: str = Field(description="PNG data URL, long side <= 1024")
    perturbation: str = Field(description="PNG data URL: |cloaked-original| x10, clipped to [0,255]")
    swap_original: str | None = Field(
        description="PNG data URL (long side <= 512, labelled SYNTHETIC): attacker's swap using the original photo"
    )
    swap_cloaked: str | None = Field(
        description="PNG data URL (long side <= 512, labelled SYNTHETIC); null if the attacker failed to detect the face"
    )
    cloaked_download: str = Field(
        description="full-resolution (as processed, <=1024 px) cloaked PNG data URL"
    )
    scores: Scores
    quality: Quality
    protected: bool | None = Field(
        description="id_sim_cloaked < tau and id_sim_original >= tau; null if the clean swap itself did not match"
    )
    note: str | None = Field(default=None, description="human-readable explanation of the verdict")
    timings: Timings
    params: Params


class JobStatus(BaseModel):
    job_id: str
    status: JobState
    queue_position: int | None = None
    progress: Progress | None = None
    error: str | None = None
    result: JobResult | None = None


class ErrorBody(BaseModel):
    detail: str
