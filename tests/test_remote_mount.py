"""Lifecycle isolation checks; no real network, processes, or mounts are changed."""

import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    "remote_mount", Path(__file__).resolve().parents[1] / "scripts/remote_mount.py"
)
mount = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mount)


class RemoteMountTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {"target": "clutterbot@127.0.0.1", "port": 2222,
                    "remote": "/home/clutterbot/robin/le_snail_ws",
                    "tag": "le-snail-" + "a" * 32}

    def test_unrelated_mount_is_never_unmounted_or_signalled(self):
        with patch.object(mount, "mount_record", return_value=("fuse.sshfs", "someone-else")), \
                patch.object(mount.subprocess, "run") as run, \
                patch.object(mount.os, "killpg") as kill:
            with self.assertRaisesRegex(RuntimeError, "unrelated mount"):
                mount.stop(self.cfg, force=True)
            run.assert_not_called()
            kill.assert_not_called()

    def test_busy_normal_stop_does_not_kill_daemon(self):
        with patch.object(mount, "owned_mount", return_value=True), \
                patch.object(mount.shutil, "which", return_value="fusermount3"), \
                patch.object(mount.subprocess, "run", return_value=subprocess.CompletedProcess([], 1)) as run, \
                patch.object(mount, "terminate") as terminate:
            with self.assertRaisesRegex(RuntimeError, "Unmount failed"):
                mount.stop(self.cfg)
            self.assertEqual(run.call_args[0][0][1], "-u")
            terminate.assert_not_called()

    def test_forced_cleanup_unmounts_before_terminating_owned_process(self):
        actions = []
        def unmount(*args, **kwargs):
            actions.append("unmount")
            self.assertEqual(args[0][1], "-uz")
            return subprocess.CompletedProcess([], 0)
        with patch.object(mount, "owned_mount", side_effect=[True, False]), \
                patch.object(mount.shutil, "which", return_value="fusermount3"), \
                patch.object(mount.subprocess, "run", side_effect=unmount), \
                patch.object(mount, "terminate", side_effect=lambda cfg: actions.append("terminate")):
            mount.stop(self.cfg, force=True)
        self.assertEqual(actions, ["unmount", "terminate"])

    def test_reused_pid_is_not_signalled(self):
        with tempfile.TemporaryDirectory() as directory:
            process = Path(directory) / "process.json"
            process.write_text(json.dumps({"pid": 12345, "start": "old"}))
            with patch.object(mount, "PROCESS", process), \
                    patch.object(mount, "process_start", return_value="new"), \
                    patch.object(mount.os, "killpg") as kill:
                mount.terminate(self.cfg)
                kill.assert_not_called()
                self.assertFalse(process.exists())

    def test_matching_pid_with_wrong_command_is_not_signalled(self):
        with tempfile.TemporaryDirectory() as directory:
            process = Path(directory) / "process.json"
            process.write_text(json.dumps({"pid": 12345, "start": "same"}))
            with patch.object(mount, "PROCESS", process), \
                    patch.object(mount, "process_start", return_value="same"), \
                    patch.object(Path, "read_bytes", return_value=b"unrelated-command\0"), \
                    patch.object(mount.os, "killpg") as kill:
                with self.assertRaisesRegex(RuntimeError, "identity differs"):
                    mount.terminate(self.cfg)
                kill.assert_not_called()

    def test_status_reads_escaped_mountinfo_without_touching_mount(self):
        content = "1 2 3:4 / /workspace\\040folder/remote_gpu rw - fuse.sshfs le-snail-test rw\n"
        with patch.object(mount, "MOUNT", Path("/workspace folder/remote_gpu")), \
                patch.object(Path, "read_text", return_value=content):
            self.assertEqual(mount.mount_record(), ("fuse.sshfs", "le-snail-test"))

    def test_ssh_reconnect_requires_noninteractive_auth_and_known_host(self):
        options = mount.ssh_options(self.cfg)
        for option in ("BatchMode=yes", "StrictHostKeyChecking=yes", "ServerAliveInterval=15"):
            self.assertIn(option, options)
        self.assertEqual(options[-2:], ["-p", "2222"])
        self.cfg["port"] = 0
        self.assertNotIn("-p", mount.ssh_options(self.cfg))

    def test_config_rejects_option_injection_and_relative_directory(self):
        for field, value in (("target", "-oProxyCommand=bad"), ("remote", "relative/path"),
                             ("port", 65536), ("remote", "/path\ncommand")):
            with self.subTest(field=field, value=value):
                with self.assertRaises(RuntimeError):
                    mount.validate({**self.cfg, field: value})

    def test_remote_exec_quotes_arguments_and_directory(self):
        cfg = {**self.cfg, "remote": "/remote/it's a folder"}
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(mount, "STATE", Path(directory)), \
                    patch.object(mount, "config", return_value=cfg), \
                    patch.object(mount.sys, "argv", ["remote_mount.py", "exec", "--", "echo", "$(id); hi"]), \
                    patch.object(mount.subprocess, "call", return_value=0) as call:
                self.assertEqual(mount.main(), 0)
                command = call.call_args[0][0]
                self.assertEqual(command[-2], self.cfg["target"])
                self.assertEqual(command[-1], "cd -- '/remote/it'\"'\"'s a folder' && exec echo '$(id); hi'")


if __name__ == "__main__":
    unittest.main()
