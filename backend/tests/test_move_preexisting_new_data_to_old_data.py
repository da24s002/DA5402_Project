import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pipeline_management')))
from move_preexisting_new_data_to_old_data import main as merge_main

def test_merge_new_data(tmp_path, monkeypatch):
    # Setup: create old and new npy files
    old_dir = tmp_path / "old"
    new_dir = tmp_path / "new"
    old_dir.mkdir()
    new_dir.mkdir()
    arr1 = np.ones((2, 3))
    arr2 = np.zeros((1, 3))
    np.save(old_dir / "class1.npy", arr1)
    np.save(new_dir / "class1.npy", arr2)
    # Patch config paths
    monkeypatch.setattr("move_preexisting_new_data_to_old_data.old_npy_file_path", str(old_dir) + os.sep)
    monkeypatch.setattr("move_preexisting_new_data_to_old_data.new_npy_file_path", str(new_dir) + os.sep)
    # Run merge
    merge_main()
    # Check merged file shape
    merged = np.load(old_dir / "class1.npy")
    assert merged.shape == (3, 3)
    # New file should be deleted
    assert not (new_dir / "class1.npy").exists()