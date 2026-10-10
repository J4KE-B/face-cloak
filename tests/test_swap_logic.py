"""Weight-free tests of swap.py: face picking with hints, the restricted unpickler, pinned-commit check."""

from __future__ import annotations

import pickle
import pickletools
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest

from facecloak import swap
from facecloak.paths import WEIGHTS_DIR


def F(x1, y1, x2, y2, score=0.9):
    return SimpleNamespace(bbox=np.array([x1, y1, x2, y2], float), det_score=score)


def test_pick_face_default_hint_and_gate():
    big, small = F(0, 0, 100, 100, 0.7), F(150, 0, 190, 40, 0.95)
    assert swap.pick_face([small, big]) is big  # largest
    assert swap.pick_face([small, big], min_score=0.6) is small  # SimSwap: highest score
    assert swap.pick_face([F(0, 0, 10, 10, 0.5)], min_score=0.6) is None
    assert swap.pick_face([small, big], [145, -5, 195, 45]) is small  # bbox hint -> nearest, not largest
    kps = np.array([[160, 10], [180, 10], [170, 20], [162, 30], [178, 30]], float)
    assert swap.pick_face([small, big], kps) is small  # kps hint
    assert swap.pick_face([small, big], [400, 400, 440, 440]) is None  # outside the gate: no silent fallback
    assert swap.pick_face([], [0, 0, 1, 1]) is None
    with pytest.raises(ValueError):
        swap.pick_face([big], [1, 2, 3])


def test_faces_in_fallback_and_aliases():
    class App:
        def __init__(self):
            self.det_model = SimpleNamespace(input_size=(640, 640))
            self.sizes = []

        def get(self, bgr):
            self.sizes.append(self.det_model.input_size)
            return [] if self.det_model.input_size == (640, 640) else [F(0, 0, 1, 1)]

    app = App()
    assert len(swap.faces_in(app, np.zeros((4, 4, 3), np.uint8))) == 1
    assert app.sizes == [(640, 640), (320, 320)] and app.det_model.input_size == (640, 640)
    assert swap._faces is swap.faces_in and swap._largest is swap.largest_face


def test_swapresult_target_bbox_defaults_to_none():
    r = swap.SwapResult(None, False, 0.0, None)
    assert r.target_bbox is None


# ----------------------------------------------------------------------------- restricted unpickler
def _stack_global_pickle(module: str, name: str) -> bytes:
    """Protocol-4 pickle that only pushes the global (no REDUCE: nothing is ever called)."""
    m, n = module.encode(), name.encode()
    return b"\x80\x04" + b"\x8c" + bytes([len(m)]) + m + b"\x8c" + bytes([len(n)]) + n + b"\x93" + b"."


def test_restricted_unpickler_blocks_dotted_attribute_walk():
    # harmless stand-in for os.system: os.getcwd, reached through a torch module that imports os
    data = _stack_global_pickle("torch.nn.modules.module", "torch.os.getcwd")
    # the stock unpickler really does resolve the dotted path (proving the attack path exists) ...
    import os

    assert pickle.loads(data) is os.getcwd
    # ... the restricted one refuses, before resolving anything
    with pytest.raises(pickle.UnpicklingError, match="blocked global"):
        swap._RestrictedUnpickler(__import__("io").BytesIO(data)).load()


@pytest.mark.parametrize(
    "module,name",
    [("os", "system"), ("builtins", "eval"), ("torch.nn.modules.linear", "Linear.__init__"),
     ("torch.nn.modules.evil", "Linear"), ("torch.nn.modules.", "Linear"), ("collections", "OrderedDict.fromkeys")],
)
def test_restricted_unpickler_blocks_everything_off_list(module, name):
    import io

    with pytest.raises(pickle.UnpicklingError):
        swap._RestrictedUnpickler(io.BytesIO(_stack_global_pickle(module, name))).load()


def test_restricted_unpickler_allows_listed_pair():
    import io
    from collections import OrderedDict

    data = _stack_global_pickle("collections", "OrderedDict")
    assert swap._RestrictedUnpickler(io.BytesIO(data)).load() is OrderedDict
    assert not any("." in n for _, n in swap._ALLOWED_GLOBALS)


def test_allow_list_matches_real_checkpoint_statically():
    """Static pickletools disassembly (nothing executed) of the real file: every GLOBAL is allow-listed."""
    arc = WEIGHTS_DIR / "simswap" / "arcface_checkpoint.tar"
    if not arc.exists():
        pytest.skip("arcface checkpoint not present")
    with zipfile.ZipFile(arc) as z:
        data = z.read("archive/data.pkl")
    found, strings = set(), []
    for op, arg, _ in pickletools.genops(data):
        if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE"):
            strings.append(arg)
        elif op.name == "GLOBAL":
            m, n = arg.split(" ", 1)
            found.add((m, n))
        elif op.name == "STACK_GLOBAL":
            found.add((strings[-2], strings[-1]))
    assert found and found <= swap._ALLOWED_GLOBALS
    assert swap._ALLOWED_GLOBALS - found <= {("builtins", "set")}  # py3 spelling of the py2 name in the file


# ----------------------------------------------------------------------------- pinned commit
def test_simswap_commit_must_match_pin(tmp_path, monkeypatch):
    import subprocess

    repo = tmp_path / "repo"
    (repo / "models").mkdir(parents=True)
    (repo / "models" / "fs_networks.py").write_text("")
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
           "PATH": __import__("os").environ["PATH"], "HOME": str(tmp_path)}
    for cmd in (["init", "-q"], ["add", "."], ["commit", "-q", "-m", "x"]):
        subprocess.run(["git", "-C", str(repo), *cmd], check=True, env=env, capture_output=True)
    with pytest.raises(RuntimeError, match="expected pinned"):
        swap._import_simswap(repo)
    with pytest.raises(RuntimeError, match="cannot verify"):
        swap._verify_simswap_commit(tmp_path / "nope")
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True, env=env).stdout.strip()
    monkeypatch.setattr(swap, "SIMSWAP_COMMIT", head)
    swap._verify_simswap_commit(repo)
