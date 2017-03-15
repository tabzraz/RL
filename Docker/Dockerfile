# FROM tensorflow/tensorflow:latest-gpu-py3
FROM nvidia/cuda:8.0-cudnn5-devel

MAINTAINER Tabish Rashid

RUN apt-get update
RUN apt-get install -y software-properties-common python-software-properties
RUN add-apt-repository ppa:jonathonf/python-3.6
RUN apt-get update
RUN apt-get install -y python3.6
RUN apt-get install -y python3-pip
RUN rm /usr/bin/python
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip

RUN pip install --upgrade pip

RUN apt-get install -y vim screen htop

# Needed for some gym dependencies
# RUN apt-get update
# RUN apt-get install -y --no-install-recommends xvfb libav-tools xorg-dev libsdl2-dev swig cmake git
RUN apt-get install -y --no-install-recommends xvfb libav-tools xorg-dev swig cmake git

#3 > 2
#RUN rm /usr/bin/python
#RUN ln -s /usr/bin/python3 /usr/bin/python

RUN pip install gym

RUN pip install gym[all]

RUN pip install pygame

RUN pip install h5py

RUN pip install scikit-image
RUN pip install matplotlib

# PyTorch
RUN pip install https://s3.amazonaws.com/pytorch/whl/cu80/torch-0.1.10.post2-cp35-cp35m-linux_x86_64.whl 
RUN pip install torchvision

RUN pip install imageio

RUN pip install tensorboard_logger

WORKDIR "/home"

RUN git clone https://github.com/tabzraz/RL.git
# RUN git clone https://github.com/tabzraz/ML.git
# RUN pip install -e ML/

# CTS Lib
RUN git clone https://github.com/tabzraz/SkipCTS.git
RUN pip install -e SkipCTS/python

CMD ["/bin/bash"]
