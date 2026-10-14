#!/usr/bin/env python3
"""Run on the laptop host: transfer sources, preflight the GPU host, build Docker."""

import argparse
from datetime import datetime, timezone
import io
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import uuid

import remote_mount


INCLUDE = ("AGENTS.md", "HANDOFF.md", "GLOSSARY.md", "README.md", "pyproject.toml", ".gitignore",
           ".dockerignore", "configs", "docker", "docs", "firmware", "scripts", "src", "tests")
SKIP = {"__pycache__", ".pytest_cache", "hardware.local.json"}


def source_archive(root):
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:gz") as archive:
        for item in INCLUDE:
            base = root / item
            paths = sorted(base.rglob("*")) if base.is_dir() else [base]
            for path in paths:
                relative = path.relative_to(root)
                if any(part in SKIP for part in relative.parts) or path.suffix == ".pyc":
                    continue
                if path.is_symlink():
                    raise RuntimeError(f"Refusing to package a symlink: {relative}")
                if path.is_file():
                    archive.add(path, arcname=str(relative), recursive=False)
    return data.getvalue()


# Run remotely only after a complete upload. Check every destination before copying
# anything, so an existing modified workspace is not silently overwritten.
INSTALL = r'''
from pathlib import Path
import shutil
import sys
stage, root = map(Path, sys.argv[1:])
files = sorted(p for p in stage.rglob('*') if p.is_file())
conflicts = []
for source in files:
    dest = root / source.relative_to(stage)
    parents = [dest, *dest.parents]
    if any(p.is_symlink() for p in parents):
        conflicts.append(str(dest) + ' (symlink)')
    elif dest.exists() and (not dest.is_file() or dest.read_bytes() != source.read_bytes()):
        conflicts.append(str(dest))
    elif any(p.exists() and not p.is_dir() for p in dest.parents):
        conflicts.append(str(dest) + ' (parent is not a directory)')
if conflicts:
    print('Existing remote files differ. No project files were replaced:', flush=True)
    print('\n'.join(conflicts), flush=True)
    print('Uploaded snapshot retained at: ' + str(stage), flush=True)
    sys.exit(2)
for source in files:
    dest = root / source.relative_to(stage)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        shutil.copy2(source, dest)
print(f'Workspace ready: {root} ({len(files)} source files checked)', flush=True)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-eula", action="store_true",
                        help="Accept the Isaac Sim EULA and run asset fetch + simulation smoke test after build")
    args = parser.parse_args()
    cfg = remote_mount.config()
    root = remote_mount.ROOT
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    remote_stage = cfg["remote"].rstrip("/") + "/.setup/incoming-" + stamp + "-" + uuid.uuid4().hex[:8]
    remote_mount.STATE.mkdir(mode=0o700, exist_ok=True)
    log_path = remote_mount.STATE / f"setup-{stamp}.log"
    ssh = ["ssh", *remote_mount.ssh_options(cfg), cfg["target"]]

    with log_path.open("wb") as log:
        def run(command, data=None):
            print(f"Remote: {command}", flush=True)
            # The upload is small; communicate avoids pipe deadlocks and records all
            # output. Build output is streamed so long downloads remain visible.
            if data is not None:
                result = subprocess.run([*ssh, command], input=data, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, timeout=180)
                log.write(result.stdout)
                log.flush()
                sys.stdout.buffer.write(result.stdout)
                sys.stdout.flush()
                code = result.returncode
            else:
                with subprocess.Popen([*ssh, command], stdin=subprocess.DEVNULL,
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as process:
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        sys.stdout.buffer.write(line)
                        sys.stdout.flush()
                    code = process.wait()
            if code:
                raise RuntimeError(f"Remote step exited {code}. Log: {log_path}")

        print(f"Setup log: {log_path}", flush=True)
        quoted_stage = shlex.quote(remote_stage)
        quoted_root = shlex.quote(cfg["remote"])
        run(f"umask 077; mkdir -p -- {quoted_stage} && tar -xzf - -C {quoted_stage}",
            source_archive(root))
        run(f"python3 - {quoted_stage} {quoted_root}", INSTALL.encode())
        run(f"cd -- {quoted_root} && bash scripts/remote_preflight.sh")
        run(f"cd -- {quoted_root} && docker build -t le-snail:isaac-2.3.2 -f docker/Dockerfile .")
        if args.accept_eula:
            # Exercise the smoke test's first import before fetching assets or launching Kit.
            import_check = "from isaaclab.app import AppLauncher; import flatdict; print('Isaac Lab launcher import passed')"
            run(f"cd -- {quoted_root} && ACCEPT_EULA=Y bash scripts/container.sh -c {shlex.quote(import_check)}")
            run(f"cd -- {quoted_root} && ACCEPT_EULA=Y bash scripts/container.sh scripts/fetch_asset.py")
            run(f"cd -- {quoted_root} && ACCEPT_EULA=Y bash scripts/container.sh "
                "scripts/smoke_sim.py --headless --device cuda:0")
        else:
            print("Build complete. Asset fetch/simulation not run; see README license and smoke-test steps.")
    print(f"Finished. Local log: {log_path}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.TimeoutExpired) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("Interrupted. Check the remote host before restarting a long build.", file=sys.stderr)
        sys.exit(130)
