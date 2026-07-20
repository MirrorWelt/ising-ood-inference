import os

from ising_ood.data.discovery import (
    discover_temperature_files,
    parse_model_type_from_name,
    parse_class_count_from_name,
)


def test_parse_model_type_from_name():
    assert parse_model_type_from_name("regression_cnn_10classes.pth") == "cnn"
    assert parse_model_type_from_name("regression_improved_cnn_50classes.pth") == "improved_cnn"
    assert parse_model_type_from_name("unrelated_file.pth") is None


def test_parse_class_count_from_name():
    assert parse_class_count_from_name("regression_gnn_16classes.pth") == 16
    assert parse_class_count_from_name("no_count_here.pth") is None


def test_discover_temperature_files(tmp_path):
    for beta, cls in [(0.1, 1), (1.0, 1), (10.0, 2)]:
        fname = f"tensor_coupling_dataset_beta_{beta}_class{cls}_samples_100.pt"
        (tmp_path / fname).write_bytes(b"placeholder")

    index = discover_temperature_files(str(tmp_path))
    assert set(index.keys()) == {1, 2}
    assert len(index[1]) == 2
    # T = 1/beta, sorted ascending in T => beta=10 (T=0.1) then beta=1 (T=1) etc. for class 1
    Ts_class1 = [e["T"] for e in index[1]]
    assert Ts_class1 == sorted(Ts_class1)