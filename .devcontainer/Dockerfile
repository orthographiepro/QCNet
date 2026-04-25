# .devcontainer/Dockerfile
FROM mcr.microsoft.com/devcontainers/miniconda:0-3

# Download rust for av2-api
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && export PATH=$HOME/.cargo/bin:$PATH && rustup default nightly

# Copy the environment file to a temporary location in the container
COPY environment.yml /tmp/conda-tmp/

# Update the base conda environment with your dependencies
RUN umask 0002 \
    && /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml \
    && conda clean -afy

# Install additional python packages
RUN conda run -n base pip install rtpt gpustat

# Get additional runtime libraries
# ... to do so, first get the key for yarn so apt may upgrade
RUN rm -f /etc/apt/sources.list.d/yarn.list
# RUN apt-key adv --recv-keys --keyserver hkp://keyserver.ubuntu.com 62D54FD4003F6525 
RUN apt-get update  \
    && apt-get install -y libgl1-mesa-dev libglib2.0-0
