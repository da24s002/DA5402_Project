import os
import pytest
import sys
# from ../pipeline_management/scan_file_count import get_count_img_in_category
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pipeline_management')))
from scan_file_count import get_count_img_in_category

def test_get_count_img_in_category_counts_png(tmp_path):
    # Setup: create a temp directory with some png and non-png files
    (tmp_path / "img1.png").write_text("fake")
    (tmp_path / "img2.png").write_text("fake")
    (tmp_path / "img3.jpg").write_text("fake")
    # Should count only PNG files
    assert get_count_img_in_category(str(tmp_path)) == 2

def test_get_count_img_in_category_empty(tmp_path):
    # Empty directory
    assert get_count_img_in_category(str(tmp_path)) == 0