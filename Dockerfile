FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu18.04

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ARG PYTHON_VERSION=3.8.6

ARG FORCE_CUDA="1"
ENV FORCE_CUDA=${FORCE_CUDA}

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    build-essential \
    bzip2 \
    curl \
    g++ \
    git \
    git-lfs \
    graphviz \
    libbz2-dev \
    libffi-dev \
    libgdbm-dev \
    libgdcm-tools \
    libgl1-mesa-dev \
    libhdf5-dev \
    liblzma-dev \
    libncurses5-dev \
    libnss3-dev \
    libreadline-dev \
    libsqlite3-dev \
    libssl-dev \
    libopenblas-dev \
    openmpi-bin \
    ssh \
    wget \
    zlib1g-dev \
    tmux \
    htop \
    && \
    apt-get clean && \
    apt-get autoremove -yqq && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN cd /opt && \
    wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
    tar xzf Python-${PYTHON_VERSION}.tgz && rm -r Python-${PYTHON_VERSION}.tgz && \
    cd /opt/Python-${PYTHON_VERSION} && \
    ./configure --enable-optimizations && \
    make -j $(nproc) && make install -j $(nproc)

RUN ln -s /usr/local/bin/pip3 /usr/local/bin/pip && \
    ln -s /usr/local/bin/python3 /usr/local/bin/python

RUN pip install --upgrade pip

# install requirements
COPY /requirements.txt /workspace/requirements.txt
RUN pip3 install -r /workspace/requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

WORKDIR /workspace

# jupyter setup
RUN jupyter nbextension enable --py widgetsnbextension