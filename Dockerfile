# use devel version not runtime for onnxruntime-gpu
FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel 
# FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime


#set shell 
SHELL ["/bin/bash", "-c"]
#set colors
ENV BUILDKIT_COLORS=run=green:warning=yellow:error=red:cancel=cyan
#start with root user
USER root
#expect build-time argument
ARG HOST_USER_GROUP_ARG
RUN groupadd -g 999 appuser && \
    groupadd -g $HOST_USER_GROUP_ARG hostgroup && \
    useradd --create-home --shell /bin/bash -u 999 -g appuser appuser && \
    echo 'appuser:admin' | chpasswd && \
    usermod -aG sudo,hostgroup,plugdev,video,adm,cdrom,dip,dialout appuser && \
    cp /etc/skel/.bashrc /home/appuser/  


#basic dependencies for everything and ROS
USER root
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive\
    apt-get install -y\
    netbase\
    git\
    build-essential\    
    wget\
    curl\
    gdb\
    lsb-release\
    gdb\
    udev\
    acl \
    libcurl4-openssl-dev \
    libtiff-dev


RUN rm -rf /var/lib/apt/lists/*
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive\
    apt-get install -y\
    sudo \
    python3-wstool\
    python3-tk\
    python3-pip\
    python-is-python3 \
    libgl1-mesa-glx \
    libglib2.0-0


USER root
RUN pip install --no-cache-dir \
    ipykernel \
    gitpython>=3.1.30 \
    matplotlib>=3.3 \
    numpy>=1.23.5 \
    opencv-python>=4.1.1 \
    Pillow>=10.3.0 \
    psutil \
    PyYAML>=5.3.1 \
    requests>=2.32.2 \
    scipy>=1.4.1 \
    thop>=0.1.1 \
    torch>=1.8.0 \
    torchvision>=0.9.0 \
    tqdm>=4.66.3 \
    ultralytics>=8.2.34 \
    pandas>=1.1.4 \
    seaborn>=0.11.0 \
    setuptools>=70.0.0 \
    roboflow 

# for onnx export
RUN pip install onnxruntime-gpu --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/ 

# for tensorrt export
RUN pip install tensorrt==10.7.0


# Install C++ development dependencies for TorchScript
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -y \
    cmake \
    g++ \
    libtorch-dev \
    libopencv-dev \
    ninja-build


# Clean up apt cache
RUN rm -rf /var/lib/apt/lists/*


USER appuser
ENTRYPOINT ["bash"]



