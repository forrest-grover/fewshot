"""Walking image folders (fewshot.folders): hidden and system entries, unreadable files."""

import pytest

from fewshot.folders import class_folders, image_files, read_or_empty
from tests.conftest import BLUE, RED, png, write_images


def _messy(folder):
    """A dataset with one real class among hidden and system folders and files."""
    for label in ("cat", ".ipynb_checkpoints", "__MACOSX", "@eaDir"):
        write_images(folder / label, RED, 2)
    (folder / "cat" / "._0.png").write_bytes(b"AppleDouble junk")
    (folder / "cat" / ".hidden.png").write_bytes(png(BLUE))
    write_images(folder / "cat" / ".ipynb_checkpoints", RED, 1)


def test_folder_walks_skip_hidden_and_system_entries(tmp_path):
    """The folder helpers leave out hidden and system entries and honour shots."""
    _messy(tmp_path)
    assert list(class_folders(tmp_path)) == ["cat"]
    assert [f.name for f in class_folders(tmp_path)["cat"]] == ["0.png", "1.png"]
    assert [f.name for f in class_folders(tmp_path, shots=1)["cat"]] == ["0.png"]
    assert [f.name for f in image_files(tmp_path / "cat", recursive=True)] == ["0.png", "1.png"]


def test_import_skips_system_folders_in_the_library_and_cli(ws, run, tmp_path):
    """Neither the library nor `fewshot import` makes classes of __MACOSX and the like."""
    _messy(tmp_path / "data")
    t = ws.create("lib", embedder="fake:color")
    assert t.add_examples_from_dir(tmp_path / "data", shots=1) == 1
    assert t.class_names() == ["cat"]
    assert run("create", "cli", "--embedder", "fake:color").exit_code == 0
    assert run("import", "cli", tmp_path / "data").exit_code == 0
    assert run("show", "cli", "--json").output.count('"name": "cat"') == 1
    assert ".ipynb" not in run("show", "cli").output


def test_add_examples_from_dir_reports_an_unreadable_file(ws, tmp_path):
    """The library import names the file it couldn't read, after adding the others."""
    write_images(tmp_path / "cat", RED, 1)
    (tmp_path / "cat" / "1.png").write_bytes(b"not an image")
    t = ws.create("t", embedder="fake:color")
    with pytest.raises(ValueError, match=r"1\.png"):
        t.add_examples_from_dir(tmp_path)
    assert len(t.config.examples) == 1


def test_an_unreadable_file_reads_as_empty(tmp_path):
    """A file that can't be read gives no bytes (and so an unreadable image), not an exception."""
    assert read_or_empty(tmp_path) == b""  # a folder
