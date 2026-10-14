#!/usr/bin/env python3
"""Manage this workspace's SSHFS mount without storing credentials (Linux)."""

import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import uuid


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / ".remote-mount"
MOUNT = ROOT / "remote_gpu"
CONFIG = STATE / "config.json"
PROCESS = STATE / "process.json"
LOG = STATE / "sshfs.log"


def fail(message):
    raise RuntimeError(message)


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def config():
    if not CONFIG.exists():
        fail("Run 'setup' first.")
    value = json.loads(CONFIG.read_text())
    validate(value)
    return value


def validate(value):
    # A target is a hostname, SSH alias, or user@hostname, never SSH options.
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@:\[\]-]*", value["target"]):
        fail("Use an SSH alias, hostname, or user@hostname for the target.")
    if type(value["port"]) is not int or not 0 <= value["port"] <= 65535:
        fail("Port must be 1–65535, or 0 to use SSH configuration.")
    path = value["remote"]
    if not path.startswith("/") or any(c in path for c in "\r\n\0"):
        fail("Use an absolute remote directory without control characters.")
    if not re.fullmatch(r"le-snail-[0-9a-f]{32}", value["tag"]):
        fail("Invalid mount identity in configuration.")


def mount_record():
    # Reading mountinfo never touches an unresponsive FUSE filesystem.
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        left, right = line.split(" - ", 1)
        raw = left.split()[4]
        path = re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), raw)
        if path == str(MOUNT):
            fields = right.split()
            return fields[0], fields[1]
    return None


def owned_mount(cfg):
    record = mount_record()
    if record and record != ("fuse.sshfs", cfg["tag"]):
        fail(f"Refusing to touch an unrelated mount at {MOUNT}: {record}")
    return bool(record)


def process_start(pid):
    try:
        # Field 22; comm (field 2) may itself contain spaces or parentheses.
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return fields[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def owned_pid(cfg):
    if not PROCESS.exists():
        return None
    info = json.loads(PROCESS.read_text())
    pid = info["pid"]
    if type(pid) is not int or pid <= 1:
        fail("Invalid recorded process ID; refusing to signal it.")
    if not info.get("start") or process_start(pid) != info["start"]:
        return None  # Includes exited processes and reused PIDs.
    try:
        argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        if f"fsname={cfg['tag']}".encode() not in argv or os.getpgid(pid) != pid:
            fail("Recorded process identity differs; refusing to signal it.")
    except (FileNotFoundError, ProcessLookupError):
        return None
    return pid


def terminate(cfg):
    for sig, wait in ((signal.SIGTERM, 3), (signal.SIGKILL, 2)):
        pid = owned_pid(cfg)
        if pid is None:
            PROCESS.unlink(missing_ok=True)
            return
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pid, sig)
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            if owned_pid(cfg) is None:
                PROCESS.unlink(missing_ok=True)
                return
            time.sleep(0.1)
    fail("SSHFS did not exit. No replacement was started; inspect the process/log.")


def ssh_options(cfg):
    options = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
               "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=15",
               "-o", "ServerAliveCountMax=3"]
    if cfg["port"]:
        options += ["-p", str(cfg["port"])]
    return options


def stop(cfg, force=False):
    if owned_mount(cfg):
        unmount = shutil.which("fusermount3") or shutil.which("fusermount")
        if not unmount:
            fail("Install sshfs/fuse3 first: sudo apt install sshfs")
        try:
            result = subprocess.run([unmount, "-uz" if force else "-u", str(MOUNT)],
                                    timeout=10, check=False)
        except subprocess.TimeoutExpired:
            fail("Unmount timed out. Close files/shells in remote_gpu, then use 'clean'.")
        if result.returncode or owned_mount(cfg):
            fail("Unmount failed. Close files/shells in remote_gpu; use 'clean' for a stale mount.")
    terminate(cfg)
    print("Stopped. Remote files, the tunnel, configuration, and logs are preserved.")


def start(cfg):
    if owned_mount(cfg):
        if owned_pid(cfg):
            print(f"Already mounted at {MOUNT}. Use 'restart' to reconnect explicitly.")
            return
        fail("A stale mount exists. Run 'clean', then 'start'.")
    if owned_pid(cfg):
        fail("SSHFS is running without a mount. Run 'clean', then 'start'.")
    for name in ("ssh", "sshfs"):
        if not shutil.which(name):
            fail(f"Missing {name}. On Ubuntu: sudo apt install openssh-client sshfs")
    if not Path("/dev/fuse").exists():
        fail("/dev/fuse is unavailable. Run this in your laptop's host terminal, not this sandbox.")
    if MOUNT.is_symlink():
        fail("The mount directory must not be a symlink.")
    MOUNT.mkdir(mode=0o700, exist_ok=True)
    if any(MOUNT.iterdir()):
        fail("The local remote_gpu directory is not empty; refusing to hide its files.")
    remote = shlex.quote(cfg["remote"])
    # No password handling: SSH uses the user's key/configuration/agent.
    result = subprocess.run(["ssh", *ssh_options(cfg), cfg["target"],
                             f"mkdir -p -- {remote} && test -w {remote}"], timeout=25)
    if result.returncode:
        fail("SSH failed. Verify the tunnel, host key, and key/ssh-agent authentication in your terminal.")
    command = ["sshfs", "-f", *ssh_options(cfg), "-o", f"fsname={cfg['tag']}",
               "-o", "reconnect,idmap=user,dir_cache=no,entry_timeout=0,attr_timeout=0",
               f"{cfg['target']}:{cfg['remote']}", str(MOUNT)]
    with LOG.open("ab", buffering=0) as log:
        log.write(f"\nStarting mount at {time.ctime()}\n".encode())
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=log, start_new_session=True, close_fds=True)
    save(PROCESS, {"pid": process.pid, "start": process_start(process.pid)})
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if process.poll() is not None:
                fail(f"SSHFS exited. Read {LOG}")
            if owned_mount(cfg):
                print(f"Mounted {cfg['target']}:{cfg['remote']} at {MOUNT}")
                print(f"Keepalives and reconnect enabled. Log: {LOG}")
                return
            time.sleep(0.2)
        fail(f"Mount timed out. Read {LOG}")
    except BaseException:
        stop(cfg, force=True)
        raise


def setup(args):
    if mount_record() or (CONFIG.exists() and owned_pid(config())):
        fail("Stop/clean the current mount before changing its configuration.")
    target = args.target or input("SSH alias or user@host (e.g. robin@127.0.0.1): ").strip()
    port = args.port
    if port is None:
        port = int(input("Tunnel port [0 = use SSH config/default]: ").strip() or "0")
    remote = args.remote or input("Absolute remote folder (e.g. /home/robin/projects/le_snail): ").strip()
    cfg = {"target": target, "port": port, "remote": remote,
           "tag": "le-snail-" + uuid.uuid4().hex}
    validate(cfg)
    save(CONFIG, cfg)
    print(f"Saved non-secret settings to {CONFIG}")
    login = ["ssh"] + (["-p", str(port)] if port else []) + [target]
    print("First verify the server fingerprint and login in your own terminal:")
    print("  " + shlex.join(login))
    print("For unattended reconnect, configure a key or load your key into ssh-agent.")
    print("Then run: python3 scripts/remote_mount.py start")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    init = sub.add_parser("setup", help="Save host/port/folder; never asks for a password")
    init.add_argument("--target")
    init.add_argument("--port", type=int)
    init.add_argument("--remote")
    for name in ("start", "status", "stop", "clean"):
        sub.add_parser(name)
    restart = sub.add_parser("restart", help="Unmount and stop this SSHFS process before starting")
    restart.add_argument("--force", action="store_true", help="Lazy-unmount stale/busy mount; open I/O can fail")
    execute = sub.add_parser("exec", help="Execute argv on the GPU host in the configured remote folder")
    execute.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    os.umask(0o077)
    STATE.mkdir(mode=0o700, exist_ok=True)
    # Lock lifecycle operations. Never hold the lock over long remote jobs.
    with (STATE / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.action == "setup":
            setup(args)
            return 0
        cfg = config()
        if args.action == "start":
            start(cfg)
        elif args.action in ("stop", "clean"):
            stop(cfg, force=args.action == "clean")
        elif args.action == "restart":
            stop(cfg, force=args.force)
            start(cfg)
        elif args.action == "status":
            mounted, pid = owned_mount(cfg), owned_pid(cfg)
            print(f"Mount: {MOUNT}\nMounted: {mounted}\nSSHFS PID: {pid}\nLog: {LOG}")
            print("Status checks local mount/process state, not remote responsiveness.")
            return 0 if mounted and pid else 1
    if args.action == "exec":
        command = args.command
        if command[:1] == ["--"]:
            command = command[1:]
        if not command:
            fail("Provide a remote command after exec --")
        remote_command = f"cd -- {shlex.quote(cfg['remote'])} && exec {shlex.join(command)}"
        return subprocess.call(["ssh", *ssh_options(cfg), cfg["target"], remote_command])
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, OSError, ValueError, KeyError, subprocess.TimeoutExpired) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)
