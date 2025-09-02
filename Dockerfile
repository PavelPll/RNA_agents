FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# 1. Installer les dépendances système
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    build-essential \
    cmake \
    libopenblas-dev \
    libssl-dev \
    curl \
    clang \
 && rm -rf /var/lib/apt/lists/*
# apt install -y clang

# 2. Installer OpenMM (via conda-forge)
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh \
 && bash miniconda.sh -b -p /opt/conda \
 && rm miniconda.sh

ENV PATH="/opt/conda/bin:${PATH}"
RUN bash -c "source ~/.bashrc"
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda create -n drfold python=3.10.4
RUN conda run -n drfold conda install -c conda-forge openmm
RUN conda run -n drfold pip install torch --index-url https://download.pytorch.org/whl/cu121
RUN conda run -n drfold pip install scipy # bcbio-gff

# Set working directory
WORKDIR /opt/drfold2

### end Dockefile ###

# RUN pip install biopython==1.80
# RUN conda run -n drfold pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# RUN mkdir file_exchange && mkdir file_exchange/fasta_input && mkdir file_exchange/pdb_output

# COPY drfold2 /opt
# COPY drfold2/Arena /opt/drfold2/Arena
# RUN ls -al /opt/drfold2
# RUN cd /opt/drfold2/Arena && make Arena



