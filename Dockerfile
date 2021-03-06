## Dockerfile for project development



## Base image
FROM ubuntu:20.04



## Working directory inside the container
WORKDIR '/rob_app'



## Environment variables

### General variables
ENV TIMEZONE America/Mexico_City
ENV JUPYTERLAB_VERSION 3.1.0
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV DEBIAN_FRONTEND noninteractive

### Dependencies and packages
ENV DEB_BUILD_DEPS="build-essential \
    python3-dev \
    python3-pip \
    python3-setuptools \
    software-properties-common \
    libgit2-dev \
    dirmngr \
    libgmp3-dev \
    libmpfr-dev"
ENV DEB_PACKAGES="sudo \
    nano \
    vim \
    less \
    git \
    curl \
    wget \
    htop"
ENV PIP_PACKAGES="numpy \
    scipy \
    matplotlib \
    pandas \
    seaborn \
    sympy \
    cvxpy \
    cvxopt \
    pytest"



## Commands execution

### Timezone and additional configuration
RUN apt-get update && \
    export $DEBIAN_FRONTEND && \
    echo $TIMEZONE > /etc/timezone && \
    apt-get install -y tzdata

### Installing basic dependencies and package manager
RUN apt-get update && \
    apt-get install -y $DEB_BUILD_DEPS $DEB_PACKAGES && \
    pip3 install --upgrade pip

### Installing jupyterlab and configuring base password
RUN groupadd myuser
RUN useradd myuser -g myuser -m -s /bin/bash
RUN echo 'myuser ALL=(ALL:ALL) NOPASSWD:ALL' | (EDITOR='tee -a' visudo)
RUN echo 'myuser:qwerty' | chpasswd
RUN pip3 install jupyter jupyterlab==$JUPYTERLAB_VERSION

### Configuring jupyterlab
USER root
RUN sudo mkdir -p rob_app/test_dir && \
    sudo touch rob_app/test_dir/test_file.txt
RUN sudo mkdir -p rob_app/.jupyterlab/user-settings/@jupyterlab/apputils-extension/ && \
    sudo echo '{ "theme":"JupyterLab Dark" }' > themes.jupyterlab-settings ## Enable dark mode

### Hashing base password
USER myuser
RUN jupyter notebook --generate-config && \
    sed -i "s/# c.NotebookApp.password = .*/c.NotebookApp.password = u'sha1:115e429a919f:21911277af52f3e7a8b59380804140d9ef3e2380'/" ~/.jupyter/jupyter_notebook_config.py

### Installing pip packages
RUN pip3 install $PIP_PACKAGES



## Default commands
ENTRYPOINT ["/usr/local/bin/jupyter", "lab", "--ip=0.0.0.0", "--no-browser"]
