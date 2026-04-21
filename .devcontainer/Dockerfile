# .devcontainer/Dockerfile
FROM mcr.microsoft.com/devcontainers/miniconda:0-3

# Copy the environment file to a temporary location in the container
COPY environment.yml /tmp/conda-tmp/

# Update the base conda environment with your dependencies
RUN umask 0002 && /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml \
    && conda clean -afy

# Download av2-api
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh && export PATH=$HOME/.cargo/bin:$PATH && rustup default nightly
RUN pip install git+https://github.com/argoverse/av2-api#egg=av2

# For Dataset Download 
RUN conda install s5cmd -c conda-forge

# Additional Requirements
RUN pip install rtpt gpustat
