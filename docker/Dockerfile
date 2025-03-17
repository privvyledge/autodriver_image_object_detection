#ARG ULTRALYTICS_VERSION=latest
ARG ULTRALYTICS_VERSION=v8.3.91

#ARG IMAGE_NAME="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"
ARG IMAGE_NAME="nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04"
FROM $IMAGE_NAME

# Set up the shell
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
ARG DEBCONF_NONINTERACTIVE_SEEN=true
ENV TZ=America/New_York
ENV HOME=/root

# Set environment variables
# Avoid DDP error "MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1 library" https://github.com/pytorch/pytorch/issues/37377
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    MKL_THREADING_LAYER=GNU \
    OMP_NUM_THREADS=1

# Install linux packages
# g++ required to build 'tflite_support' and 'lap' packages, libusb-1.0-0 required for 'tflite_support' package
# libsm6 required by libqxcb to create QT-based windows for visualization; set 'QT_DEBUG_PLUGINS=1' to test in docker
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    sudo gcc git zip unzip wget curl htop libgl1 libglib2.0-0 libpython3-dev gnupg g++ libusb-1.0-0 libsm6 \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -yq --no-install-recommends tzdata && \
    ln -fns /usr/share/zoneinfo/${TZ} /etc/localtime && echo $TZ > /etc/timezone && \
    dpkg-reconfigure -f noninteractive tzdata && \
    rm -rf /var/lib/apt/lists/*

# Security updates
# https://security.snyk.io/vuln/SNYK-UBUNTU1804-OPENSSL-3314796
RUN apt upgrade --no-install-recommends -y openssl tar

RUN sudo apt-get update -y && DEBIAN_FRONTEND="noninteractive" sudo apt-get install -y --no-install-recommends \
    sudo \
    git \
    curl \
    wget \
    less \
    zstd \
    udev \
    unzip \
    build-essential \
    apt-transport-https \
    openssh-server libv4l-0 libv4l-dev v4l-utils binutils xz-utils bzip2 lbzip2 \
    ca-certificates libegl1 \
    software-properties-common \
    lsb-release \
    gnupg2 \
    cmake \
    ccache \
    pkg-config \
    swig \
    g++ \
    libpython3-dev \
    python3-dev \
    python3 \
    python3-venv \
    python3.10 \
    python3.10-venv \
    python3-pip \
#    python3-setuptools \
#    python3-numpy \
#    python3-rosdep \
#    python3-matplotlib \
#    python3-opencv \
#    python3-pil \
    python3-yaml \
    python3-tk \
    python3-pyqt5 \
    python3-bloom \
#    libopencv-dev \
    libssl-dev \
    libusb-1.0-0-dev \
    libgtk-3-dev \
    libglfw3-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libgoogle-glog-dev \
    qtcreator && \
    sudo rm -rf /var/lib/apt/lists/*

## Install python3.10
RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    python3-venv \
    libpython3-dev \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    libpython3.10-dev && \
    rm -rf /var/lib/apt/lists/*
#RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends \
#    software-properties-common && \
#    sudo add-apt-repository -y ppa:deadsnakes/ppa && \
#    apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends \
#    python3.10 \
#    python3.10-dev \
#    python3.10-venv \
#    libpython3.10-dev && \
#    rm -rf /var/lib/apt/lists/*

# Update pip
RUN python3.10 -m pip install --upgrade pip wheel

# Set python3.10 as the default python
#RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
#    update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
#    echo "alias python3=python3.10" >> /etc/bash.bashrc

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Install ROS 2 Humble
ARG ROS_DISTRO=humble

# Install ROS for non-ROS base packages.
ENV LANG=en_US.UTF-8

RUN apt update && DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends \
    locales && \
    locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 && \
    export LANG=en_US.UTF-8 && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository -y universe && \
    apt update && apt-get install -y --no-install-recommends curl && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
    apt update && \
    apt-get remove -y python3-catkin-pkg python3-rospkg python3-rosdistro && \
    apt purge python-rosdep2 python3-rosdep2 -y && \
    apt update && DEBIAN_FRONTEND="noninteractive" apt install -y \
    python3-rosdep \
    ros-humble-desktop \
    ros-humble-perception \
    ros-dev-tools \
    ros-humble-rmw-cyclonedds-cpp && \
    rm -rf /var/lib/apt/lists/* && \
    . /opt/ros/humble/setup.bash && \
    sudo rosdep init && \
    rosdep update && \
    rm -rf /var/lib/apt/lists/*


# ROS fundamentals
RUN --mount=type=cache,target=/var/cache/apt \
        sudo apt-get update -y && DEBIAN_FRONTEND="noninteractive" sudo apt-get install -y --no-install-recommends \
        devscripts \
        dh-make \
        fakeroot \
        libxtensor-dev \
        python3-bloom \
        python3-colcon-common-extensions \
        python3-pip \
        python3-pybind11 \
        python3-pytest-cov \
        python3-rosdep \
        python3-rosinstall-generator \
        python3-vcstool \
        python3-colcon-mixin \
        python3-flake8-docstrings \
        python3-pip \
        python3-pytest-cov \
        ros-dev-tools \
        python3-flake8-blind-except \
        python3-flake8-builtins \
        python3-flake8-class-newline \
        python3-flake8-comprehensions \
        python3-flake8-deprecated \
        python3-flake8-import-order \
        python3-flake8-quotes \
        python3-pytest-repeat \
        python3-pytest-rerunfailures \
        quilt && \
    sudo rm -rf /var/lib/apt/lists/*

# ROS Python fundamentals. The -U flag will update numpy but downstream commands will install 1.26.6
RUN python3.10 -m pip install \
        flake8-blind-except \
        flake8-builtins \
        flake8-class-newline \
        flake8-comprehensions \
        flake8-deprecated \
        flake8-docstrings \
        flake8-import-order \
        flake8-quotes \
#        "numpy==1.26.4" \
#        "setuptools==65.7.0" \
        matplotlib \
        pandas \
        rosbags && \
    sudo rm -rf /var/lib/apt/lists/*

## Install CUDA
#RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
#    sudo dpkg -i cuda-keyring_1.1-1_all.deb && \
#    rm cuda-keyring_1.1-1_all.deb && \
#    sudo apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends \
#    cuda-${CUDA} \
#    cuda-compat-${CUDA} \
#    cuda-toolkit-${CUDA} \
#    cuda-command-line-tools-${CUDA} \
#    cuda-cudart-${CUDA} \
#    cuda-libraries-${CUDA} \
#    cuda-nvml-dev-${CUDA} \
#    cuda-sanitizer-${CUDA} \
#    libcublas-${CUDA} \
#    libcudnn${cudnn_major_version} \
#    libcusparse-${CUDA} \
#    libnpp-${CUDA} && \
#    apt-get clean && rm -rf /var/lib/apt/lists/*
#
## Install cuDNN
#RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends \
#    libcudnn${cudnn_major_version} && \
#    sudo apt-mark hold libcudnn${cudnn_major_version} && \
#    DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends libcudnn${cudnn_major_version}-dev && \
#    sudo apt-mark hold libcudnn${cudnn_major_version}-dev && \
#    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install PyTorch
RUN sudo apt purge python3-sympy -y && python3.10 -m pip install "torch==2.5.1" "torchvision==0.20.1" --index-url https://download.pytorch.org/whl/cu124

# Install tensorrt
RUN python3.10 -m pip install tensorrt-cu12 tensorrt-lean-cu12 tensorrt-dispatch-cu12

# Install Ultralytics
ARG ULTRALYTICS_VERSION
RUN sudo apt-get update && \
    DEBIAN_FRONTEND="noninteractive" sudo apt-get install -y --no-install-recommends \
    git-lfs gcc git zip curl htop libgl1 libglib2.0-0 libpython3-dev gnupg g++ libusb-1.0-0 && \
    mkdir -p ${HOME}/.config/Ultralytics && cd ${HOME}/.config/Ultralytics && \
    echo ${ULTRALYTICS_VERSION} && \
    wget https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.ttf && \
    wget https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.Unicode.ttf && \
    cd ${HOME} && git clone https://github.com/ultralytics/ultralytics -b ${ULTRALYTICS_VERSION} && \
    cd ultralytics && \
    python3.10 -m pip install --upgrade pip wheel "setuptools==70.0.0" && \
    cd ${HOME}/ultralytics/ && python3.10 -m pip install -e ".[export]" && \
    python3.10 -m pip install "numpy==1.26.4" && \
    sudo rm -rf /var/lib/apt/lists/*

ARG ROS_VERSION="ROS2"
ARG ROS_DISTRO="humble"
ENV ROS_DISTRO=${ROS_DISTRO}
ENV ROS_ROOT=/opt/ros/${ROS_DISTRO}
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# Create a ROS 2 workspace
ARG BUILD_HOME=/ros2_ws
RUN mkdir -p ${BUILD_HOME}/src
WORKDIR ${BUILD_HOME}

RUN cd ${BUILD_HOME}/src && \
    git clone https://github.com/privvyledge/autodriver_image_object_detection.git

# Install gscam dependencies
RUN sudo apt-get update -y && DEBIAN_FRONTEND="noninteractive" sudo apt-get install -y --no-install-recommends \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    gstreamer1.0-x \
    gstreamer1.0-alsa \
    gstreamer1.0-gl \
    gstreamer1.0-gtk3 \
    gstreamer1.0-qt5 \
    gstreamer1.0-pulseaudio \
    libgstreamer-plugins-base1.0-dev \
    libglib2.0-dev \
    libgstrtspserver-1.0-dev \
    gstreamer1.0-rtsp && \
    sudo rm -rf /var/lib/apt/lists/*

RUN sudo apt-get update -y && DEBIAN_FRONTEND="noninteractive" sudo apt-get install -y --no-install-recommends \
    ros-${ROS_DISTRO}-tf2-tools \
    ros-${ROS_DISTRO}-tf-transformations \
    ros-${ROS_DISTRO}-rqt-tf-tree \
    ros-${ROS_DISTRO}-tf2-geometry-msgs \
    ros-${ROS_DISTRO}-gscam \
    ros-${ROS_DISTRO}-vision-msgs \
    ros-${ROS_DISTRO}-nav-msgs \
    ros-${ROS_DISTRO}-derived-object-msgs \
    ros-${ROS_DISTRO}-shape-msgs && \
    sudo apt purge python3-transforms3d -y && \
    python3.10 -m pip install "transforms3d==0.4.2" && \
    sudo rm -rf /var/lib/apt/lists/*

# Build ROS2 packages
ARG SKIPPED_ROSDEP_KEYS="cmake_modules pacmod3_msgs  python3-open3d open3d \
    librealsense2 realsense2_camera libopencv-dev  libopencv-contrib-dev libopencv-imgproc-dev python-opencv python3-opencv"
RUN sudo apt update && \
    rosdep update && \
    source /opt/ros/${ROS_DISTRO}/setup.bash && \
    rosdep install --from-paths src --ignore-src -r -y -q --skip-keys "${SKIPPED_ROSDEP_KEYS}"

RUN source /opt/ros/${ROS_DISTRO}/setup.bash && \
    colcon build --symlink-install --event-handlers console_direct+ --base-paths src --cmake-args \
    ' -Wno-dev' ' --no-warn-unused-cli' \
    ' -DCMAKE_BUILD_TYPE=Release' ' -DCMAKE_EXPORT_COMPILE_COMMANDS=ON' \
    ' -DCMAKE_CXX_FLAGS="-Wl,--allow-shlib-undefined"'

ENV RCUTILS_COLORIZED_OUTPUT=1 \
    RCUTILS_CONSOLE_OUTPUT_FORMAT="[{severity} {time}] [{name}]: {message} ({function_name}() at {file_name}:{line_number})" \
    _colcon_cd_root=${ROS_ROOT}

ENV PYTHONPATH="${PYTHONPATH}:/usr/local/lib"

RUN echo 'alias build="colcon build --symlink-install --event-handlers console_direct+"' >> /etc/bash.bashrc && \
    echo "source /usr/share/colcon_cd/function/colcon_cd.sh" >> /etc/bash.bashrc && \
    echo "source /opt/ros/humble/setup.bash" >> /etc/bash.bashrc && \
    echo "source /ros2_ws/install/setup.bash" >> /etc/bash.bashrc

## Todo: remove the lines below in Release/production version
RUN sudo apt update && sudo apt install gedit cheese nautilus net-tools iputils-ping nano -y

CMD ["/bin/bash"]