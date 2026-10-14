"""Check transfer contents and conflict handling without contacting a server."""

import io
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import setup_remote
sys.path.pop(0)


class RemoteSetupTests(unittest.TestCase):
    def test_archive_excludes_private_local_files_and_mount(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("README.md", "scripts/train.py", "configs/policy.json",
                         "configs/hardware.local.json", "src/__pycache__/x.pyc",
                         ".aws/credentials", ".remote-mount/config.json", "remote_gpu/private"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("test")
            with tarfile.open(fileobj=io.BytesIO(setup_remote.source_archive(root))) as archive:
                self.assertEqual(set(archive.getnames()),
                                 {"README.md", "scripts/train.py", "configs/policy.json"})

    def install(self, stage, root):
        return subprocess.run([sys.executable, "-", str(stage), str(root)],
                              input=setup_remote.INSTALL, text=True, capture_output=True)

    def test_existing_change_aborts_before_copying_any_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root, stage = Path(directory) / "remote", Path(directory) / "stage"
            root.mkdir()
            stage.mkdir()
            (root / "z.txt").write_text("remote work")
            (stage / "z.txt").write_text("local version")
            (stage / "a.txt").write_text("new file")
            result = self.install(stage, root)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertEqual((root / "z.txt").read_text(), "remote work")
            self.assertFalse((root / "a.txt").exists())

    def test_identical_workspace_can_be_reinstalled(self):
        with tempfile.TemporaryDirectory() as directory:
            root, stage = Path(directory) / "remote", Path(directory) / "stage"
            stage.mkdir()
            (stage / "README.md").write_text("project")
            for _ in range(2):
                result = self.install(stage, root)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual((root / "README.md").read_text(), "project")

    def test_remote_symlink_cannot_redirect_source_install(self):
        with tempfile.TemporaryDirectory() as directory:
            root, stage, outside = [Path(directory) / name for name in ("remote", "stage", "outside")]
            root.mkdir()
            (stage / "src").mkdir(parents=True)
            outside.mkdir()
            (root / "src").symlink_to(outside, target_is_directory=True)
            (stage / "src/file.py").write_text("code")
            result = self.install(stage, root)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertFalse((outside / "file.py").exists())
