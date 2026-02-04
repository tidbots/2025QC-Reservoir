FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

SHELL ["/bin/bash", "-c"]
ARG DEBIAN_FRONTEND=noninteractive

# ---- Nvidia runtime config ----
ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics
ENV __NV_PRIME_RENDER_OFFLOAD=1
ENV __GLX_VENDOR_LIBRARY_NAME=nvidia

# ---- Locale and timezone ----
RUN apt update && apt install -y --no-install-recommends \
  locales \
  language-pack-ja-base language-pack-ja \
  software-properties-common tzdata \
  fonts-ipafont fonts-ipaexfont fonts-takao \
  && locale-gen ja_JP.UTF-8 && update-locale LANG=ja_JP.UTF-8 \
  && ln -fs /usr/share/zoneinfo/Asia/Tokyo /etc/localtime && dpkg-reconfigure --frontend noninteractive tzdata

ENV LANG=ja_JP.UTF-8
ENV TZ=Asia/Tokyo

# ---- Core utilities ----
RUN apt update && apt install -y --no-install-recommends \
  dirmngr \
  git build-essential iputils-ping net-tools \
  cmake g++ iproute2 gnupg2 \
  libcanberra-gtk* \
  python3-pip python3-tk python-is-python3 \
  mesa-utils x11-utils x11-apps \
  terminator xterm nano vim htop \
  gdb valgrind curl wget sudo \
  && apt clean && rm -rf /var/lib/apt/lists/*

# ---- ROS 2 Humble ----
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg

RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
> /etc/apt/sources.list.d/ros2.list

RUN apt update && apt install -y \
  ros-humble-desktop \
  ros-humble-rmw-cyclonedds-cpp \
  python3-rosdep python3-colcon-common-extensions \
  python3-vcstool python3-genpy \
  libpcap-dev libopenblas-dev \
  gstreamer1.0-tools libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-good1.0-dev \
  libportaudio2 alsa-base alsa-utils \
  portaudio19-dev pulseaudio \
  sshpass ffmpeg python3-pyaudio \
  ninja-build stow chrony \
  && apt clean && rm -rf /var/lib/apt/lists/*

# ---- Python packages ----
RUN pip install --no-cache-dir --ignore-installed \
    setuptools==59.6.0 wheel \
    rosnumpy stitching opencv-python rospkg numpy==1.26.4 \
    pillow torch torchvision transformers==4.40.0 \
    vosk sounddevice spacy==3.7.4 jinja2==3.0.3 \
    SpeechRecognition openai pydub openai-whisper \
    scipy>=1.12 sparray scikit-learn reservoirpy

# ---- Optional: tmux ----
RUN apt update && apt install -y automake autoconf pkg-config libevent-dev libncurses5-dev bison && \
    git clone https://github.com/tmux/tmux.git && cd tmux && git checkout tags/3.2 && \
    sh autogen.sh && ./configure && make -j8 && make install && cd .. && rm -rf tmux

# ---- Additional pumas_nav2 deps ----
RUN apt update && apt install -y \
    ros-humble-tf2-ros \
    ros-humble-tf2-eigen \
    ros-humble-tf2-geometry-msgs \
    ros-humble-visualization-msgs \
    ros-humble-cartographer ros-humble-cartographer-ros \
    libpcl-dev ros-humble-pcl-ros ros-humble-pcl-conversions \
    ros-humble-nav2-map-server ros-humble-nav2-lifecycle-manager \
    ros-humble-depth-image-proc \
    && apt clean && rm -rf /var/lib/apt/lists/*

# ---- assets that rarely change ----
COPY assets/terminator_config /etc/skel/.config/terminator/config
COPY assets/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# ---- User-specific setup ----
ARG UID
ARG GID
ARG USER_NAME
ARG GROUP_NAME
ARG PASSWORD

RUN groupadd -g $GID $GROUP_NAME && \
    useradd -m -s /bin/bash -u $UID -g $GID -G sudo $USER_NAME && \
    echo $USER_NAME:$PASSWORD | chpasswd && \
    echo "$USER_NAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    cp -r /etc/skel/. /home/$USER_NAME/ && chown -R $USER_NAME:$GROUP_NAME /home/$USER_NAME

USER ${USER_NAME}

RUN sudo rosdep init && rosdep update

RUN git config --global user.email "you@example.com" && git config --global user.name "Your Name"

# ---- Docker install ----
RUN curl -fsSL https://get.docker.com -o ~/get-docker.sh && \
    sh ~/get-docker.sh && sudo usermod -aG docker $USER_NAME && \
    sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose && \
    sudo chmod +x /usr/local/bin/docker-compose

# ---- Workspaces ----
# ros2_ws
RUN  mkdir -p /home/$USER_NAME/ros2_ws
COPY ./ros2_ws /home/$USER_NAME/ros2_ws
RUN /bin/bash -c "sudo apt-get update && rosdep install --from-paths ~/ros2_ws/src/ --ignore-src --rosdistro humble -r -y"
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && cd ~/ros2_ws && colcon build"

RUN echo "source /home/$USER_NAME/ros2_ws/install/setup.bash" >> ~/.bashrc

# ---- Frequently changed args ----
# RMW_IMPLEMENTATION config 
COPY assets/cyclonedds_profile.xml /home/$USER_NAME/.config/cyclonedds_profile.xml 

ARG NETWORK_IF
ARG WORKSPACE_DIR
ARG DOCKER_DIR
ARG ROBOT_NAME
ARG ROS_IP
ARG DOMAIN_ID

RUN { \
    echo "network_if=${NETWORK_IF}"; \
    echo "export TARGET_IP=$ROS_IP"; \
    echo 'if [ -n "$TARGET_IP" ]; then export ROS_IP=$TARGET_IP; fi'; \
    echo "export ROS_DOMAIN_ID=$DOMAIN_ID"; \
    echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp"; \
    echo "export CYCLONEDDS_URI=/home/$USER_NAME/.config/cyclonedds_profile.xml"; \
    echo "source /opt/ros/humble/setup.bash"; \
    echo "export ROBOT_NAME=${ROBOT_NAME}"; \
    echo "export WORKSPACE_DIR=${WORKSPACE_DIR}"; \
    echo "export DOCKER_DIR=${DOCKER_DIR}"; \
    echo "alias sim_mode='export ROS_LOCALHOST_ONLY=1'"; \
    echo "export PS1=\"\[\033[44;1;37m\]<hsrc>\[\033[0m\]\w$ \""; \
} >> ~/.bashrc

# ---- Python path ----
ENV PYTHONPATH "/opt/ros/humble/lib/python3.10/site-packages:${PYTHONPATH}"

ENTRYPOINT ["/entrypoint.sh"]
CMD ["terminator"]

