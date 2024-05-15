ARG PYTORCH="1.9.0"
ARG CUDA="10.2"
ARG CUDNN="7"
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# To fix GPG key error when running apt-get update
RUN rm /etc/apt/sources.list.d/cuda.list \
    && rm /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# Install system dependencies for opencv-python
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 git

# Install mmcv
ARG MMCV="1.3.9"
RUN pip install -U openmim
RUN mim install mmcv-full==${MMCV}
# RUN pip install mmcv-full==${MMCV} -f https://download.openmmlab.com/mmcv/dist/cu${CUDA}/torch${PYTORCH}/index.html
RUN pip install mmcv-full==1.3.9 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.9.0/index.html


# Verify the installation
RUN python -c 'import mmcv;print(mmcv.__version__)'

# Install ViTPose
WORKDIR /
RUN git clone https://github.com/ViTAE-Transformer/ViTPose.git
WORKDIR /ViTPose
RUN pip install -v -e .
RUN pip install timm==0.4.9 einops
# RUN pip install git+https://github.com/NVIDIA/apex.git

# Clean
WORKDIR /workspace
RUN apt-get clean \
    && rm -rf /var/lib/apt/lists/*
