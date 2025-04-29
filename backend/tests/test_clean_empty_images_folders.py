# tests/test_clean_empty_images_folders.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pipeline_management')))
from clean_empty_images_folders import delete_empty_subfolders

def test_delete_empty_subfolders(tmp_path):
    # Create two subfolders, one empty, one with a file
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    non_empty_dir = tmp_path / "non_empty"
    non_empty_dir.mkdir()
    (non_empty_dir / "file.txt").write_text("data")
    # Run function
    delete_empty_subfolders(str(tmp_path))
    # Assert empty folder is deleted, non-empty remains
    assert not empty_dir.exists()
    assert non_empty_dir.exists()