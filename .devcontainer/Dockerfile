FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# ── System dependencies ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget \
        bzip2 \
        ca-certificates \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Miniforge ────────────────────────────────────────────────────────────────
ENV PATH="/opt/miniforge/bin:${PATH}"

RUN wget -q \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" \
        -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /opt/miniforge \
    && rm /tmp/miniforge.sh \
    && conda init bash

# ── Conda environment ────────────────────────────────────────────────────────
COPY .devcontainer/environment.yml /tmp/environment.yml

RUN conda env create -f /tmp/environment.yml \
    && conda clean -afy

# Activate marp env by default for all shells and processes
ENV PATH="/opt/miniforge/envs/marp/bin:${PATH}"
RUN echo "conda activate marp" >> /root/.bashrc

# ── Install marp as editable package ──────────────────────────────────────────
COPY pyproject.toml README.md /tmp/marp_src/
COPY marp/ /tmp/marp_src/marp/
COPY third_party/ /tmp/marp_src/third_party/
RUN pip install -e /tmp/marp_src/third_party/zigzag \
    && pip install -e "/tmp/marp_src[dev]" \
    && pip cache purge

# ── Runtime configuration ────────────────────────────────────────────────────
# Use non-interactive Matplotlib backend (no display needed)
ENV MPLBACKEND=Agg
