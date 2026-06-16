import json

import numpy as np

from pysimplemask.core.io import load_pixel_list, text_to_array


def test_text_to_array_ints():
    out = text_to_array("[1, 2], [3, 4]")
    assert out.tolist() == [1, 2, 3, 4]


def test_text_to_array_floats():
    out = text_to_array("1.5 2.5", dtype=np.float64)
    assert np.allclose(out, [1.5, 2.5])


def test_load_pixel_list_csv(tmp_path):
    p = tmp_path / "pts.csv"
    p.write_text("1,2\n3,4\n")
    out = load_pixel_list(str(p))
    assert out.tolist() == [[1, 2], [3, 4]]


def test_load_pixel_list_txt_space(tmp_path):
    p = tmp_path / "pts.txt"
    p.write_text("1 2\n3 4\n")
    out = load_pixel_list(str(p))
    assert out.tolist() == [[1, 2], [3, 4]]


def test_load_pixel_list_json(tmp_path):
    p = tmp_path / "pts.json"
    p.write_text(json.dumps({"Bad pixels": [{"Pixel": [1, 2]}, {"Pixel": [3, 4]}]}))
    out = load_pixel_list(str(p))
    assert out.tolist() == [[1, 2], [3, 4]]
