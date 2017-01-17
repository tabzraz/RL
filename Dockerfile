FROM tensorflow/tensorflow:latest-gpu-py3

MAINTAINER Tabish Rashid

RUN pip install tqdm

RUN pip install tflearn

# Needed for some gym dependencies
RUN apt-get update
RUN apt-get install -y --no-install-recommends xvfb libav-tools xorg-dev libsdl2-dev swig cmake git

RUN pip install gym

RUN pip install gym[all]

RUN pip install pygame

RUN pip install h5py

WORKDIR "/home"

RUN git clone https://github.com/tabzraz/RL.git
RUN git clone https://github.com/tabzraz/ML.git

RUN pip install -e ML/

# 3 > 2
RUN rm /usr/bin/python
RUN ln -s /usr/bin/python3 /usr/bin/python

CMD ["/bin/bash"]
