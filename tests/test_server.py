"""Backend API tests with a stub engine: no model weights are loaded, everything runs instantly."""

from __future__ import annotations

import base64
import io
import logging
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from server import imaging
from server.app import client_ip, create_app
from server.engine import StubEngine
from server.jobs import RateLimiter
from server.schema import CONSENT_NOTICE, JobStatus
from server.settings import Settings

KEY = "test-passcode-0123456789"
H = {"X-Demo-Key": KEY}


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


class GatedEngine(StubEngine):
    """Blocks inside cloak() until released, to observe queued/running states."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.entered = threading.Event()
        self.release = threading.Event()

    def cloak(self, img_rgb, face, eps, steps, mode, on_step):
        on_step(1, steps)
        self.entered.set()
        assert self.release.wait(10)
        return super().cloak(img_rgb, face, eps, steps, mode, on_step)


def jpeg_bytes(w: int = 128, h: int = 96, fmt: str = "JPEG", **kw) -> bytes:
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, fmt, **kw)
    return buf.getvalue()


LOOPBACK = (
    "127.0.0.1"  # socket peer of the local cloudflared tunnel: the only peer whose CF headers are trusted
)


def make_client(engine=None, clock=None, peer=LOOPBACK, **settings_kw):
    kw = {"demo_key": KEY, "queue_cap": 50, "job_ttl_s": 600.0, **settings_kw}
    app = create_app(
        Settings(**kw), engine_factory=lambda: engine or StubEngine(), clock=clock or FakeClock()
    )
    return TestClient(app, client=(peer, 50000))


def cf(ip: str) -> dict:
    return {**H, "CF-Connecting-IP": ip}


def png_of(data_url: str) -> np.ndarray:
    assert data_url.startswith("data:image/png;base64,")
    raw = base64.b64decode(data_url.split(",", 1)[1])
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))


def post_job(c, photo=None, headers=None, data=None, files=None):
    f = files or {"photo": ("a.jpg", photo or jpeg_bytes(), "image/jpeg")}
    d = {"target_id": "t1", **(data or {})}
    return c.post("/api/jobs", files=f, data=d, headers=H if headers is None else headers)


def wait_status(c, job_id, want, timeout=10.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = c.get(f"/api/jobs/{job_id}", headers=H)
        if r.json()["status"] in want:
            return r.json()
        time.sleep(0.02)
    raise AssertionError(f"job did not reach {want}: {r.json()}")


def test_health_open_and_shape():
    with make_client() as c:
        j = c.get("/health").json()
        assert j["status"] == "ok" and j["models_loaded"] is True and j["device"] == "cpu"
        assert j["default_steps"] == 30 and j["max_eps"] == 16 and j["tau"] == pytest.approx(0.144)
        assert j["queue_length"] == 0


def test_refuses_to_start_without_key():
    with pytest.raises(RuntimeError, match="DEMO_KEY"):
        create_app(Settings(), engine_factory=StubEngine)
    create_app(Settings(insecure_dev=True), engine_factory=StubEngine)  # explicit dev opt-in is fine


@pytest.mark.parametrize("key", ["", "short", "a" * 11])
def test_short_key_refused(key):
    if key == "":  # empty == unset -> the missing-key error
        with pytest.raises(RuntimeError, match="DEMO_KEY"):
            create_app(Settings(demo_key=key or None), engine_factory=StubEngine)
        return
    with pytest.raises(RuntimeError, match="too short"):
        create_app(Settings(demo_key=key), engine_factory=StubEngine)
    with pytest.raises(RuntimeError, match="too short"):  # dev mode does not excuse a weak passcode
        create_app(Settings(demo_key=key, insecure_dev=True), engine_factory=StubEngine)
    create_app(Settings(demo_key="a" * 12), engine_factory=StubEngine)


def test_dev_mode_needs_no_key():
    app = create_app(Settings(insecure_dev=True), engine_factory=StubEngine)
    with TestClient(app) as c:
        assert c.get("/api/targets").status_code == 200


@pytest.mark.parametrize("headers", [{}, {"X-Demo-Key": "wrong"}, {"X-Demo-Key": ""}])
def test_auth_401(headers):
    with make_client() as c:
        assert c.get("/api/targets", headers=headers).status_code == 401
        assert post_job(c, headers=headers).status_code == 401
        assert c.get("/api/jobs/abc", headers=headers).status_code == 401


def test_targets():
    with make_client() as c:
        r = c.get("/api/targets", headers=H)
        assert r.status_code == 200 and r.json()[0]["id"] == "t1"
        assert set(r.json()[0]) == {"id", "name", "thumbnail", "source", "license"}


def test_413_streamed_oversize_photo():
    with make_client() as c:
        big = b"\xff\xd8\xff" + b"\0" * (6 * 1024 * 1024)
        assert post_job(c, big).status_code == 413
        huge = b"\xff\xd8\xff" + b"\0" * (30 * 1024 * 1024)  # far above the whole-body cap
        assert post_job(c, huge).status_code == 413


def test_415_non_image():
    with make_client() as c:
        r = post_job(c, b"hello, this is not an image at all")
        assert r.status_code == 415 and "JPEG" in r.json()["detail"]
        r = post_job(c, b"\xff\xd8\xff" + b"garbage" * 20)  # right magic bytes, undecodable
        assert r.status_code == 415


def test_decompression_bomb_rejected():
    with make_client() as c:
        buf = io.BytesIO()
        Image.new("1", (7000, 7000)).save(buf, "PNG")  # 49 MP, a few KB on the wire
        assert len(buf.getvalue()) < 5 * 1024 * 1024
        r = post_job(c, buf.getvalue())
        assert r.status_code == 413 and "24 MP" in r.json()["detail"]


@pytest.mark.parametrize("eps", [1, 17, "abc"])
def test_422_bad_eps(eps):
    with make_client() as c:
        assert post_job(c, data={"eps": str(eps)}).status_code == 422


def test_422_bad_steps_mode_target():
    with make_client() as c:
        assert post_job(c, data={"steps": "201"}).status_code == 422
        assert post_job(c, data={"mode": "turbo"}).status_code == 422
        assert post_job(c, data={"target_id": "nope"}).status_code == 422


def test_400_no_face():
    with make_client(StubEngine(detect_ok=False)) as c:
        r = post_job(c)
        assert r.status_code == 400 and "No face" in r.json()["detail"]


def test_exif_orientation_and_resize():
    img = Image.fromarray(np.zeros((100, 200, 3), np.uint8))
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 270 -> portrait
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif)
    assert imaging.decode_upload(buf.getvalue()).shape[:2] == (200, 100)
    big = imaging.decode_upload(jpeg_bytes(2000, 1000, "PNG"))
    assert max(big.shape[:2]) == 1024
    assert imaging.decode_upload(jpeg_bytes(100, 100, "WEBP")).shape[:2] == (100, 100)


def test_job_lifecycle_queued_running_done():
    eng = GatedEngine()
    with make_client(eng) as c:
        r1 = post_job(c, data={"steps": "10", "eps": "8"})
        assert r1.status_code == 202
        j1 = r1.json()["job_id"]
        assert eng.entered.wait(5)
        s1 = JobStatus.model_validate(c.get(f"/api/jobs/{j1}", headers=H).json())
        assert s1.status == "running" and s1.progress.stage == "cloaking" and s1.progress.total == 10
        r2 = post_job(c)
        assert r2.json()["queue_position"] == 1
        j2 = r2.json()["job_id"]
        s2 = c.get(f"/api/jobs/{j2}", headers=H).json()
        assert s2["status"] == "queued" and s2["queue_position"] == 1 and s2["result"] is None
        assert c.get("/health").json()["queue_length"] == 2
        eng.release.set()
        d1 = wait_status(c, j1, {"done"})
        wait_status(c, j2, {"done"})
        res = JobStatus.model_validate(d1).result
        assert (
            res.protected is True and res.scores.id_sim_original > res.scores.tau > res.scores.id_sim_cloaked
        )
        assert res.scores.reference == "self" and res.scores.judge == "glintr100"
        assert res.params.eps == 8 and res.params.steps == 10 and res.params.mode == "fast"
        assert res.quality.linf_255 == 8
        for k in ("original", "cloaked", "perturbation", "swap_original", "swap_cloaked", "cloaked_download"):
            assert getattr(res, k).startswith("data:image/png;base64,")
        assert res.timings.total_s >= res.timings.cloak_s >= 0


def test_reference_photo_and_robust_mode():
    with make_client() as c:
        files = {
            "photo": ("a.jpg", jpeg_bytes(), "image/jpeg"),
            "reference": ("b.png", jpeg_bytes(fmt="PNG"), "image/png"),
        }
        jid = post_job(c, files=files, data={"mode": "robust", "eps": "4"}).json()["job_id"]
        res = JobStatus.model_validate(wait_status(c, jid, {"done"})).result
        assert res.scores.reference == "uploaded" and res.params.mode == "robust"


def test_reference_without_face_400_and_tiny_image_422():
    class RefNoFace(StubEngine):
        def detect(self, img):
            return None if img.shape[1] == 80 else super().detect(img)  # reference of width 80 has no face

    with make_client(RefNoFace()) as c:
        photo = ("a.jpg", jpeg_bytes(), "image/jpeg")
        ok = {"photo": photo, "reference": ("b.png", jpeg_bytes(100, 100, "PNG"), "image/png")}
        assert post_job(c, files=ok).status_code == 202
        bad = {"photo": photo, "reference": ("b.png", jpeg_bytes(80, 100, "PNG"), "image/png")}
        r = post_job(c, files=bad)
        assert r.status_code == 400 and "reference" in r.json()["detail"]
        assert post_job(c, jpeg_bytes(40, 40)).status_code == 422


def test_verdicts_inconclusive_and_detector_failure():
    with make_client(StubEngine(clean_matches=False)) as c:
        jid = post_job(c).json()["job_id"]
        res = JobStatus.model_validate(wait_status(c, jid, {"done"})).result
        assert res.protected is None and "Inconclusive" in res.note
    with make_client(StubEngine(cloaked_undetected=True)) as c:
        jid = post_job(c).json()["job_id"]
        res = JobStatus.model_validate(wait_status(c, jid, {"done"})).result
        assert res.protected is True and res.swap_cloaked is None and res.scores.id_sim_cloaked is None


def test_worker_error_is_sanitised():
    class Boom(StubEngine):
        def swap(self, src, target_id):
            raise RuntimeError("secret internal path /home/x")

    with make_client(Boom()) as c:
        jid = post_job(c).json()["job_id"]
        st = wait_status(c, jid, {"error"})
        assert "secret" not in st["error"] and st["result"] is None


def test_ttl_expiry():
    clock = FakeClock()
    with make_client(clock=clock) as c:
        jid = post_job(c).json()["job_id"]
        assert wait_status(c, jid, {"done"})["result"] is not None
        clock.t += 599
        assert c.get(f"/api/jobs/{jid}", headers=H).json()["status"] == "done"
        clock.t += 2
        st = c.get(f"/api/jobs/{jid}", headers=H).json()
        assert st["status"] == "expired" and st["result"] is None
        clock.t += 700  # tombstone dropped too
        assert c.get(f"/api/jobs/{jid}", headers=H).status_code == 404


def test_rate_limit_6th_is_429_per_ip():
    clock = FakeClock()
    with make_client(clock=clock) as c:
        for _ in range(5):
            assert post_job(c, headers=cf("203.0.113.7")).status_code == 202
        r = post_job(c, headers=cf("203.0.113.7"))
        assert r.status_code == 429 and int(r.headers["Retry-After"]) > 0
        # another client IP is unaffected
        assert post_job(c, headers=cf("198.51.100.9")).status_code == 202
        clock.t += 3601  # window slides
        assert post_job(c, headers=cf("203.0.113.7")).status_code == 202


def test_charge_happens_before_parsing_garbage_still_costs():
    with make_client() as c:
        for _ in range(3):
            assert post_job(c, b"not an image").status_code == 415
        assert post_job(c, jpeg_bytes(40, 40)).status_code == 422  # too small
        assert post_job(c, data={"target_id": "nope"}).status_code == 422
        assert (
            post_job(c, files={"photo": ("a.jpg", b"x", "image/jpeg")}, data={"eps": "99"}).status_code == 429
        )
        # even a request that would fail form validation was charged: the 6th (valid) one is throttled
        assert post_job(c).status_code == 429


def test_charge_is_before_body_is_read():
    """A throttled client is rejected without the multipart body ever being consumed."""
    with make_client(rate_limit_per_hour=1) as c:
        assert post_job(c).status_code == 202
        r = c.post(
            "/api/jobs",
            content=b"--x\r\n" + b"\0" * 4096,
            headers={**H, "Content-Type": "multipart/form-data; boundary=x"},
        )
        assert r.status_code == 429  # decided from headers alone


def test_refund_only_on_413_and_503():
    with make_client(rate_limit_per_hour=2) as c:
        big = b"\xff\xd8\xff" + b"\0" * (6 * 1024 * 1024)
        huge = b"\xff\xd8\xff" + b"\0" * (30 * 1024 * 1024)
        for blob in (big, huge, big, huge):  # in-app 413 and Content-Length 413, each refunded
            assert post_job(c, blob).status_code == 413
        assert post_job(c).status_code == 202
        assert post_job(c).status_code == 202
        assert post_job(c).status_code == 429
    eng = GatedEngine()
    with make_client(eng, queue_cap=1, rate_limit_per_hour=3) as c:
        post_job(c, headers=cf("9.9.9.1"))
        assert eng.entered.wait(5)
        assert post_job(c, headers=cf("9.9.9.1")).status_code == 202
        for _ in range(4):  # queue full -> 503 each time, refunded each time
            assert post_job(c, headers=cf("9.9.9.1")).status_code == 503
        assert len(c.app.state.jobs.limiter._hits["9.9.9.1"]) == 2  # only the two accepted jobs stay charged
        assert len(c.app.state.jobs.global_limiter._hits["*"]) == 2
        eng.release.set()


def test_chunked_oversize_body_is_413_and_refunded():
    def body():
        yield b'--x\r\nContent-Disposition: form-data; name="photo"; filename="a.jpg"\r\n\r\n'
        for _ in range(12):  # 12 MB, no Content-Length (Transfer-Encoding: chunked)
            yield b"\0" * (1024 * 1024)

    with make_client(rate_limit_per_hour=1) as c:
        h = {**H, "Content-Type": "multipart/form-data; boundary=x"}
        r = c.post("/api/jobs", content=body(), headers=h)
        assert r.status_code == 413 and "too large" in r.json()["detail"]
        assert post_job(c).status_code == 202  # the 413 was refunded


def test_wrong_passcode_throttle_429_per_ip():
    with make_client() as c:
        bad = {"X-Demo-Key": "wrong-wrong-wrong"}
        for _ in range(20):
            assert (
                c.get("/api/targets", headers={**bad, "CF-Connecting-IP": "203.0.113.5"}).status_code == 401
            )
        # the 21st attempt is throttled - even with the right passcode, for that IP only
        r = c.get("/api/targets", headers={**cf("203.0.113.5")})
        assert r.status_code == 429 and int(r.headers["Retry-After"]) > 0
        assert c.get("/api/targets", headers={**bad, "CF-Connecting-IP": "203.0.113.5"}).status_code == 429
        assert c.get("/api/targets", headers=cf("203.0.113.6")).status_code == 200


def test_wrong_passcode_throttle_expires():
    clock = FakeClock()
    with make_client(clock=clock) as c:
        for _ in range(20):
            c.get("/api/targets", headers={"X-Demo-Key": "nope-nope-nope"})
        assert c.get("/api/targets", headers=H).status_code == 429
        clock.t += 601
        assert c.get("/api/targets", headers=H).status_code == 200


def test_cf_header_ignored_unless_peer_is_loopback():
    # direct (non-tunnel) peer forging CF-Connecting-IP cannot dodge the per-IP limit
    with make_client(peer="198.51.100.77") as c:
        for i in range(5):
            assert post_job(c, headers=cf(f"203.0.113.{i}")).status_code == 202
        assert post_job(c, headers=cf("203.0.113.99")).status_code == 429
    # through the tunnel (loopback peer) the header is the client key
    with make_client(peer=LOOPBACK) as c:
        for i in range(8):
            assert post_job(c, headers=cf(f"203.0.113.{i}")).status_code == 202


def test_client_ip_unit():
    def sc(peer, hdr=None):
        return {"client": (peer, 1), "headers": [(b"cf-connecting-ip", hdr.encode())] if hdr else []}

    assert client_ip(sc("127.0.0.1", "203.0.113.4")) == "203.0.113.4"
    assert client_ip(sc("::1", "203.0.113.4")) == "203.0.113.4"
    assert client_ip(sc("::ffff:127.0.0.1", "203.0.113.4")) == "203.0.113.4"
    assert client_ip(sc("127.0.0.1")) == "127.0.0.1"
    assert client_ip(sc("127.0.0.1", "not-an-ip")) == "127.0.0.1"
    assert client_ip(sc("198.51.100.7", "203.0.113.4")) == "198.51.100.7"  # forged header ignored
    assert client_ip({"client": None, "headers": []}) == "unknown"
    # IPv6 is keyed by /64
    assert client_ip(sc("2001:db8:1:2::1")) == client_ip(sc("2001:db8:1:2:ffff:eeee:dddd:1"))
    assert client_ip(sc("2001:db8:1:2::1")) == "2001:db8:1:2::/64"
    assert client_ip(sc("2001:db8:1:3::1")) != client_ip(sc("2001:db8:1:2::1"))
    assert client_ip(sc("127.0.0.1", "2001:db8:9:9::5")) == "2001:db8:9:9::/64"
    assert client_ip(sc("::ffff:198.51.100.7")) == "198.51.100.7"


def test_ipv6_64_shares_one_quota():
    with make_client(rate_limit_per_hour=2) as c:
        a, b = "2001:db8:1:2::1", "2001:db8:1:2:dead:beef:0:7"  # same /64
        assert post_job(c, headers=cf(a)).status_code == 202
        assert post_job(c, headers=cf(b)).status_code == 202
        assert post_job(c, headers=cf("2001:db8:1:2:1:1:1:1")).status_code == 429
        assert post_job(c, headers=cf("2001:db8:1:3::1")).status_code == 202  # other /64


def test_global_hourly_cap():
    clock = FakeClock()
    with make_client(clock=clock, global_hourly_cap=3) as c:
        for i in range(3):
            assert post_job(c, headers=cf(f"203.0.113.{i}")).status_code == 202
        r = post_job(c, headers=cf("203.0.113.200"))  # a brand-new IP, but the demo is at capacity
        assert r.status_code == 429 and "capacity" in r.json()["detail"]
        assert int(r.headers["Retry-After"]) > 0
        clock.t += 3601
        assert post_job(c, headers=cf("203.0.113.200")).status_code == 202


def test_rate_limiter_reserve_refund():
    t = [0.0]
    rl = RateLimiter(3, 100.0, lambda: t[0])
    assert rl.reserve("k", 2) is None
    assert rl.reserve("k", 2) is not None  # atomic: would exceed, nothing charged
    assert rl.reserve("k", 1) is None
    assert rl.reserve("k", 1) is not None
    rl.refund("k", 2)
    assert rl.reserve("k", 2) is None
    assert rl.reserve("k", 5) == 100  # cost alone beyond the limit


def test_step_caps_per_device():
    with make_client() as c:  # stub device = cpu: max 60
        assert c.get("/health").json()["max_steps"] == 60
        r = post_job(c, data={"steps": "61"}, headers=cf("1.1.1.1"))
        assert r.status_code == 422 and "60" in r.json()["detail"]
        assert post_job(c, data={"steps": "60"}, headers=cf("1.1.1.1")).status_code == 202

    class Cuda(StubEngine):
        device = "cuda"

    with make_client(Cuda()) as c:
        assert c.get("/health").json()["max_steps"] == 200
        assert post_job(c, data={"steps": "200"}).status_code == 202
        assert post_job(c, data={"steps": "201"}).status_code == 422  # schema ceiling


def test_robust_counts_as_two_on_cpu_but_one_on_cuda():
    with make_client() as c:  # cpu, 5/hour
        assert post_job(c, data={"mode": "robust"}).status_code == 202  # 2 used
        assert post_job(c, data={"mode": "robust"}).status_code == 202  # 4 used
        r = post_job(c, data={"mode": "robust"})  # needs 2, only 1 left
        assert r.status_code == 429 and "Robust" in r.json()["detail"]

    class Cuda(StubEngine):
        device = "cuda"

    with make_client(Cuda()) as c:
        for _ in range(5):
            assert post_job(c, data={"mode": "robust"}).status_code == 202
        assert post_job(c, data={"mode": "robust"}).status_code == 429


def test_admission_semaphore_limits_concurrent_detect():
    class Counting(StubEngine):
        def __init__(self):
            super().__init__()
            self.lock, self.cur, self.max = threading.Lock(), 0, 0

        def detect(self, img_rgb):
            with self.lock:
                self.cur += 1
                self.max = max(self.max, self.cur)
            time.sleep(0.15)
            with self.lock:
                self.cur -= 1
            return super().detect(img_rgb)

    for slots in (1, 2):
        eng = Counting()
        with make_client(eng, admission_slots=slots, global_hourly_cap=99) as c:
            assert c.app.state.admission._value == slots
            codes = []

            def go(i, c=c, codes=codes):
                codes.append(post_job(c, headers=cf(f"10.0.0.{i}")).status_code)

            ts = [threading.Thread(target=go, args=(i,)) for i in range(6)]
            [t.start() for t in ts]
            [t.join(30) for t in ts]
            assert codes == [202] * 6
            assert eng.max == slots


def test_default_admission_is_two():
    with make_client() as c:
        assert c.app.state.admission._value == 2


def test_synthetic_watermark_on_swap_outputs():
    with make_client() as c:
        jid = post_job(c).json()["job_id"]
        res = JobStatus.model_validate(wait_status(c, jid, {"done"})).result
        so, sc = png_of(res.swap_original), png_of(res.swap_cloaked)
        assert so.shape == sc.shape == (128, 128, 3)
        band = max(12, round(128 * 0.08))
        # stub clean swap is all-black: only the label text (white) can be non-zero, and only in the bottom band
        assert so[: 128 - band].max() == 0 and so[128 - band :].max() > 150
        # stub cloaked swap is all-white: the semi-transparent band darkens it, the rest is untouched
        assert (sc[: 128 - band] == 255).all() and sc[128 - band :].min() < 150
        assert (sc[128 - band :] < 255).any()
        # non-swap images are not labelled
        assert (png_of(res.original) == png_of(res.original)).all()


def test_swap_output_caps_long_side_and_stamps():
    big = np.zeros((900, 600, 3), np.uint8)
    out = imaging.swap_output(big)
    assert max(out.shape[:2]) == 512 and out.shape[:2] == (512, 341)
    assert out[-30:].max() > 150 and out[:400].max() == 0
    small = imaging.swap_output(np.zeros((100, 80, 3), np.uint8))
    assert small.shape[:2] == (100, 80)  # never upscaled


def test_self_reference_note_and_health_consent():
    with make_client() as c:
        h = c.get("/health").json()
        assert h["consent_notice"] == CONSENT_NOTICE and "consented" in CONSENT_NOTICE
        jid = post_job(c).json()["job_id"]
        res = JobStatus.model_validate(wait_status(c, jid, {"done"})).result
        assert res.scores.reference == "self" and "optimistic" in res.note and "second" in res.note
        assert res.protected is True  # behaviour unchanged
        files = {
            "photo": ("a.jpg", jpeg_bytes(), "image/jpeg"),
            "reference": ("b.png", jpeg_bytes(fmt="PNG"), "image/png"),
        }
        jid = post_job(c, files=files, headers=cf("5.5.5.5")).json()["job_id"]
        res = JobStatus.model_validate(wait_status(c, jid, {"done"})).result
        assert res.scores.reference == "uploaded" and "optimistic" not in res.note


def test_logs_do_not_contain_job_ids(caplog):
    with caplog.at_level(logging.INFO, logger="facecloak.server"), make_client() as c:
        jid = post_job(c).json()["job_id"]
        wait_status(c, jid, {"done"})
    assert "queued" in caplog.text and jid not in caplog.text


def test_dev_mode_rejects_tunnel_headers():
    app = create_app(Settings(insecure_dev=True), engine_factory=StubEngine)
    with TestClient(app) as c:
        assert c.get("/health").status_code == 200
        assert c.get("/api/targets").status_code == 200
        for h in ({"cf-ray": "abc-LHR"}, {"CF-Connecting-IP": "1.2.3.4"}):
            assert c.get("/health", headers=h).status_code == 403
            assert c.get("/api/targets", headers=h).status_code == 403
            assert post_job(c, headers=h).status_code == 403


def test_docs_only_in_unauthenticated_dev_mode():
    dev = create_app(Settings(insecure_dev=True), engine_factory=StubEngine)
    with TestClient(dev) as c:
        assert c.get("/docs").status_code == 200 and c.get("/openapi.json").status_code == 200
    for s in (Settings(demo_key=KEY), Settings(demo_key=KEY, insecure_dev=True)):
        with TestClient(create_app(s, engine_factory=StubEngine)) as c:
            assert c.get("/docs").status_code == 404 and c.get("/openapi.json").status_code == 404


def test_queue_cap_503():
    eng = GatedEngine()
    with make_client(eng, queue_cap=1) as c:
        post_job(c)
        assert eng.entered.wait(5)
        assert post_job(c).status_code == 202  # fills the single queue slot
        r = post_job(c, headers={**H, "CF-Connecting-IP": "1.1.1.1"})
        assert r.status_code == 503 and "Retry-After" in r.headers
        eng.release.set()


def test_delete_removes_job():
    with make_client() as c:
        jid = post_job(c).json()["job_id"]
        wait_status(c, jid, {"done"})
        assert c.delete(f"/api/jobs/{jid}", headers=H).status_code == 204
        assert c.get(f"/api/jobs/{jid}", headers=H).status_code == 404
        assert c.delete(f"/api/jobs/{jid}", headers=H).status_code == 404


def test_delete_cancels_queued_and_running():
    eng = GatedEngine()
    with make_client(eng) as c:
        j1 = post_job(c).json()["job_id"]
        assert eng.entered.wait(5)
        j2 = post_job(c).json()["job_id"]
        assert c.delete(f"/api/jobs/{j2}", headers=H).status_code == 204
        assert c.get("/health").json()["queue_length"] == 1
        assert c.delete(f"/api/jobs/{j1}", headers=H).status_code == 204
        eng.release.set()
        time.sleep(0.2)
        assert c.get("/health").json()["queue_length"] == 0


def preflight(c, origin):
    return c.options(
        "/api/jobs",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "x-demo-key,content-type",
        },
    )


@pytest.mark.parametrize(
    "origin,allowed",
    [
        ("https://face-cloak.vercel.app", True),
        ("https://face-cloak-git-main-j4ke-b.vercel.app", True),
        ("https://face-cloak-3k2j1h-j4ke-b.vercel.app", True),
        ("http://localhost:3000", False),  # dev mode only
        ("https://evil.com", False),
        ("https://other-project.vercel.app", False),
        ("https://face-cloak.vercel.app.evil.com", False),
        (
            "https://face-cloak-git-main-attacker.vercel.app",
            False,
        ),  # same project prefix, someone else's team
        ("https://face-cloak-x-j4ke-b.vercel.app.evil.com", False),
    ],
)
def test_cors_preflight(origin, allowed):
    with make_client(vercel_team_slug="j4ke-b") as c:
        r = preflight(c, origin)
        if allowed:
            assert r.status_code == 200
            assert r.headers["access-control-allow-origin"] == origin
            assert "x-demo-key" in r.headers["access-control-allow-headers"].lower()
            assert "access-control-allow-credentials" not in r.headers
        else:
            assert "access-control-allow-origin" not in r.headers
            assert r.status_code == 400


def test_cors_no_team_slug_means_no_preview_regex():
    s = Settings(demo_key=KEY)
    assert s.cors_origin_regex() is None
    with make_client() as c:
        assert preflight(c, "https://face-cloak.vercel.app").status_code == 200
        assert preflight(c, "https://face-cloak-git-main-j4ke-b.vercel.app").status_code == 400


def test_cors_localhost_only_in_dev_mode():
    assert "http://localhost:3000" not in Settings(demo_key=KEY).cors_origins()
    assert "http://localhost:3000" in Settings(insecure_dev=True).cors_origins()
    app = create_app(Settings(insecure_dev=True), engine_factory=StubEngine)
    with TestClient(app) as c:
        assert preflight(c, "http://localhost:3000").status_code == 200


def test_cors_env(monkeypatch):
    monkeypatch.setenv("DEMO_KEY", KEY)
    monkeypatch.setenv("DEMO_SITE", "https://demo.example.org/")
    monkeypatch.setenv("VERCEL_TEAM_SLUG", "My-Team")
    s = Settings.from_env()
    assert s.cors_origins() == ["https://demo.example.org"]
    with TestClient(create_app(s, engine_factory=StubEngine)) as c:
        assert preflight(c, "https://demo.example.org").status_code == 200
        assert preflight(c, "https://face-cloak.vercel.app").status_code == 400  # default site is replaced
        assert preflight(c, "https://face-cloak-git-x-my-team.vercel.app").status_code == 200
    monkeypatch.delenv("DEMO_SITE")
    monkeypatch.delenv("VERCEL_TEAM_SLUG")
    s = Settings.from_env()
    assert s.cors_origins() == ["https://face-cloak.vercel.app"] and s.cors_origin_regex() is None


def test_cors_env_origin_and_error_responses_carry_cors():
    with make_client(allowed_origins=["https://demo.example.org"]) as c:
        r = c.get(
            "/api/targets", headers={"Origin": "https://demo.example.org"}
        )  # 401 must still be readable by the browser
        assert r.status_code == 401 and r.headers["access-control-allow-origin"] == "https://demo.example.org"
