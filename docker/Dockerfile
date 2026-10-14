FROM nvcr.io/nvidia/isaac-sim:5.1.0
USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential cmake libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch v2.3.2 https://github.com/isaac-sim/IsaacLab.git /opt/IsaacLab
RUN ln -s /isaac-sim /opt/IsaacLab/_isaac_sim
# flatdict's legacy build imports pkg_resources, removed by setuptools 82.
# Build only this dependency without isolation, using known-compatible build tools.
RUN /isaac-sim/python.sh -m pip install setuptools==78.1.1 wheel==0.45.1 && \
    /isaac-sim/python.sh -m pip install --no-build-isolation --use-pep517 flatdict==4.0.1
# Isaac Lab runs `tabs 4` under set -e; use a terminal definition present in the base image.
RUN TERM=xterm /opt/IsaacLab/isaaclab.sh --install rsl_rl
# The upstream find -exec installer can report success after a package install failed.
# Check packages before exporting an image; importing the root module needs no GPU.
RUN /isaac-sim/python.sh -c "import isaaclab, flatdict; from importlib.util import find_spec; from importlib.metadata import version; names = ('isaaclab', 'isaaclab_assets', 'isaaclab_tasks', 'isaaclab_rl', 'isaaclab_mimic', 'isaaclab_contrib'); missing = [name for name in names if find_spec(name) is None]; assert not missing, f'Missing Isaac Lab packages: {missing}'; print({name: version(name) for name in names})"
RUN /isaac-sim/python.sh -m pip install rsl-rl-lib==3.0.1 pytest imageio imageio-ffmpeg
WORKDIR /workspace/le_snail
COPY . .
RUN /isaac-sim/python.sh -m pip install --no-deps -e .
ENTRYPOINT ["/isaac-sim/python.sh"]
