# Remote workspace through the TCP tunnel

Run `scripts/remote_mount.py` on the **local Ubuntu laptop**, as your normal user.
It exposes the GPU machine's workspace at `remote_gpu/` in this project. Files
written there are written to the remote machine. The script does not transfer the
existing project automatically, delete remote files, manage the external tunnel,
or store passwords/private keys.

The provided endpoint is `127.0.0.1:2222`, forwarding to SSH port 22. The remote
folder is `/home/clutterbot/robin/le_snail_ws`. The username `clutterbot` is inferred
from that path; replace it below if your SSH account differs. The SSH server must
allow SFTP, and the tunnel must be running whenever you use the mount.

## One-time setup

Install the local mount tools if needed:

```bash
sudo apt install sshfs openssh-client
cd /home/robin/ssd/workspaces/hackathon_ws
python3 scripts/remote_mount.py setup \
  --target clutterbot@127.0.0.1 --port 2222 \
  --remote /home/clutterbot/robin/le_snail_ws
```

An existing SSH alias also works: set `--target ALIAS --port 0` to inherit its port,
username, identity, and proxy settings from your SSH configuration.

In your own terminal, verify SSH connectivity and the server's host-key fingerprint:

```bash
ssh -p 2222 clutterbot@127.0.0.1
```

Exit the remote shell before continuing. Passwords typed into this SSH prompt stay
between you and SSH; never paste them into chat. Unattended reconnection needs SSH
key authentication. If you already use a key, load it into your SSH agent with
`ssh-add /path/to/your/private_key` and run the mount script from that same shell.
An unencrypted key already selected by your SSH configuration also works.

If you do not have a key for this machine, these commands create a **new dedicated
key** and install only its public half. Choose another filename if this one exists;
do not overwrite an existing key. Enter passphrases/passwords only in your terminal:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/le_snail_gpu
ssh-copy-id -i ~/.ssh/le_snail_gpu.pub -p 2222 clutterbot@127.0.0.1
ssh-add ~/.ssh/le_snail_gpu
```

If `ssh-add` says no agent is available, start one in that shell with
`eval "$(ssh-agent -s)"`, then repeat `ssh-add`. The agent must remain available for
reconnections. The mount script uses strict host-key checking and noninteractive
authentication; it will fail with an actionable error instead of requesting or
recording a password.

## Mount lifecycle

```bash
python3 scripts/remote_mount.py start
python3 scripts/remote_mount.py status
python3 scripts/remote_mount.py stop
python3 scripts/remote_mount.py restart
```

`start` creates the remote directory if necessary and launches SSHFS independently
of the terminal. Keepalives run every 15 seconds with three missed replies allowed;
SSHFS reconnects when the SSH endpoint becomes available again. It does not recreate
the TCP tunnel, survive a laptop reboot, or restart a crashed SSHFS process. Use
`start` after reboot and `restart --force` to recover a stale instance.

`restart` first unmounts and terminates only this script's tracked SSHFS process
group, then starts a new mount. Normal stop/restart refuses a busy mount. Close files
and move terminal working directories out of `remote_gpu/`, then retry.

For a stale/broken mount, or when you explicitly want to detach it even if busy:

```bash
python3 scripts/remote_mount.py clean
python3 scripts/remote_mount.py start
# Equivalent single command:
python3 scripts/remote_mount.py restart --force
```

Forced cleanup lazily unmounts and terminates the owned process group; open file
operations can fail and in-flight writes may be lost. Save/close files first when
possible. It never runs `pkill ssh`, kills unrelated mounts/tunnels, or recursively
deletes the mount directory. PID start times and mount identities are checked before
cleanup. Configuration and logs remain in `.remote-mount/`, which is excluded from
Git and Docker builds along with `remote_gpu/`.

`status` reads kernel mount information without accessing the remote filesystem,
so it can report a disconnected mount without hanging on directory access. It does
not prove that the remote host is currently responding. SSHFS logs are in
`.remote-mount/sshfs.log`. Reopen files after reconnecting; existing file handles may
no longer work. These behaviors follow the [SSHFS manual](https://github.com/libfuse/sshfs/blob/master/sshfs.rst).

## Transfer the project and run on the GPU

For a complete initial transfer and Docker build, run this from the laptop's host
terminal (it uses SSH directly, so an SSHFS mount is not required):

```bash
python3 scripts/setup_remote.py
```

It packages the current source, excludes credentials/local hardware configuration,
caches and mounted folders, uploads to a unique `.setup/` staging directory, and
refuses to overwrite differing files in an existing remote workspace. It then runs
the remote preflight and builds the Docker image. Logs are saved locally as
`.remote-mount/setup-TIMESTAMP.log`; staging snapshots remain remotely for inspection.
The remote user needs working Docker permissions and the NVIDIA container runtime.
Failures stop the setup at that step; no system packages or drivers are changed.

To also fetch the arm asset and run the simulation smoke test, review the Isaac Sim
license linked in the README and use `python3 scripts/setup_remote.py --accept-eula`.
This explicitly sets `ACCEPT_EULA=Y` for those container runs. Without the flag, setup
stops after the image build. A successful smoke test is not a trained crawling policy.

After `start` succeeds, files can be copied or edited through `remote_gpu/`. For the
previously prepared archive, extract its contents into the mounted directory:

```bash
python3 scripts/remote_mount.py status && \
  tar -xzf /tmp/le-snail-implementation.tar.gz --strip-components=1 -C remote_gpu
```

This copies the earlier implementation snapshot; the archive predates this mount
helper. Extraction overwrites matching project files, so use an empty remote workspace
for the initial transfer. It does not include simulator dependencies or robot assets.

**A mount provides remote files, not remote execution.** Running Python normally
from inside `remote_gpu/` still uses the laptop. Use the helper's `exec` command or
a remote terminal to run on the GPU machine:

```bash
python3 scripts/remote_mount.py exec -- nvidia-smi
python3 scripts/remote_mount.py exec -- bash scripts/remote_preflight.sh
python3 scripts/remote_mount.py exec -- docker build \
  -t le-snail:isaac-2.3.2 -f docker/Dockerfile .
```

`exec` changes into the configured directory remotely and preserves argument
boundaries. It works without an active mount. For long training jobs use `tmux` in
an interactive SSH terminal so an SSH/tunnel interruption does not end the job:

```bash
ssh -t -p 2222 clutterbot@127.0.0.1 \
  'cd /home/clutterbot/robin/le_snail_ws && tmux new-session -A -s le-snail'
```

Install `tmux` on the remote host if missing. Detach with Ctrl-B then D. Run Docker
and training against the remote host's own directory, where large caches/checkpoints
stay local to its SSD. Follow the main README for the build and simulation checks.

The current agent sandbox has no `/dev/fuse`, so mounting must be done from your
host terminal. Whether the host mount is visible inside the agent sandbox depends
on mount-namespace propagation; verify that `remote_gpu/` is visible before editing
through it. If it is not, open the mounted folder in a host-side Codex session or use
SSH-based remote commands from an environment allowed to reach the tunnel.
