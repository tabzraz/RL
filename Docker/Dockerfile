FROM nvidia/cuda:8.0-cudnn6-devel

MAINTAINER Tabish Rashid

RUN apt-get update
RUN apt-get install -y software-properties-common python-software-properties
RUN add-apt-repository ppa:jonathonf/python-3.6
RUN apt-get update
RUN apt-get install -y python3.6
RUN apt-get update
RUN apt-get install -y libpython3.6-dev
RUN apt-get install -y curl
RUN curl https://bootstrap.pypa.io/get-pip.py | python3.6

RUN rm /usr/bin/python3
RUN ln -s /usr/bin/python3.6 /usr/bin/python3
RUN ln -s /usr/bin/pip3.6 /usr/bin/pip

RUN pip install --upgrade pip

RUN apt-get install -y vim screen htop

# Fix screen Library path
RUN echo "export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64" >> ~/.bashrc

# Needed for some gym dependencies
RUN apt-get install -y --no-install-recommends xvfb libav-tools xorg-dev swig cmake git
# Doom dependencies
RUN apt-get install -y zlib1g-dev libjpeg-dev libboost-all-dev gcc libsdl2-dev wget unzip
# Mario dependency
RUN apt-get install -y fceux

# Python3 fix for libboost
RUN ln -sf /usr/lib/x86_64-linux-gnu/libboost_python-py35.so /usr/lib/x86_64-linux-gnu/libboost_python-py27.so.1.58.0

RUN pip install gym
RUN pip install gym[all]
RUN pip install pygame
RUN pip install h5py

RUN pip install vizdoom

# Mario env
RUN git clone https://github.com/ppaquette/gym-super-mario
RUN pip install -e gym-super-mario/

RUN pip install pympler

RUN pip install scikit-image
RUN pip install matplotlib

# PyTorch
RUN pip install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp36-cp36m-manylinux1_x86_64.whl 
RUN pip install torchvision

RUN pip install imageio
RUN pip install tensorboard_logger

WORKDIR "/home"

RUN git clone https://github.com/tabzraz/RL.git

CMD ["/bin/bash"]
