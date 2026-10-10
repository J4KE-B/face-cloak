
import numpy as np
import pytest

from facecloak.io import ResumableCSV, read_rgb, to_tensor, to_uint8, write_png


def test_to_tensor_and_uint8():
    pytest.importorskip("torch")
    img = np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)
    tensor = to_tensor(img)
    assert tensor.shape == (1, 3, 32, 32)
    
    img_out = to_uint8(tensor)
    assert img_out.shape == (32, 32, 3)
    assert img_out.dtype == np.uint8
    np.testing.assert_allclose(img, img_out, atol=1)

def test_read_write_png(tmp_path):
    pytest.importorskip("cv2")
    img = np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)
    p = tmp_path / "test.png"
    write_png(str(p), img)
    img_in = read_rgb(str(p))
    np.testing.assert_array_equal(img, img_in)
    
    with pytest.raises(FileNotFoundError):
        read_rgb(str(tmp_path / "missing.png"))

def test_resumable_csv(tmp_path):
    p = tmp_path / "log.csv"
    csv = ResumableCSV(str(p), key_cols=["id1", "id2"])
    
    csv.append({"id1": "A", "id2": "B", "val": 1})
    csv.append({"id1": "C", "id2": "D", "val": 2})
    
    assert csv.done_keys() == {("A", "B"), ("C", "D")}
    
    csv2 = ResumableCSV(str(p), key_cols=["id1", "id2"])
    assert csv2.done_keys() == {("A", "B"), ("C", "D")}
    
    csv2.append({"id1": "E", "id2": "F", "val": 3})
    
    with open(p, 'r') as f:
        lines = f.readlines()
        assert len(lines) == 4
        assert "id1,id2,val" in lines[0]


def test_resumable_csv_drops_truncated_last_line(tmp_path):
    path = tmp_path / "res.csv"
    path.write_text("id,val\n1,a\n2,b\n3,")  # run killed mid-write of row 3
    csv_ = ResumableCSV(str(path), key_cols=["id"])
    assert csv_.done_keys() == {("1",), ("2",)}
    csv_.append({"id": 3, "val": "c"})
    assert path.read_text() == "id,val\n1,a\n2,b\n3,c\n"


def test_write_png_atomic_overwrite_leaves_no_temp(tmp_path):
    pytest.importorskip("cv2")
    p = tmp_path / "sub" / "a.png"
    for v in (10, 20):
        write_png(str(p), np.full((8, 8, 3), v, np.uint8))
    assert read_rgb(str(p))[0, 0, 0] == 20
    assert [f.name for f in p.parent.iterdir()] == ["a.png"]
