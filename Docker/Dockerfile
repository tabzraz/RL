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

RUN pip install gym
RUN pip install gym[all]
RUN pip install pygame
RUN pip install h5py

RUN pip install pympler

RUN pip install scikit-image
RUN pip install matplotlib

# PyTorch
RUN pip install http://download.pytorch.org/whl/cu80/torch-0.1.12.post2-cp36-cp36m-linux_x86_64.whl
RUN pip install torchvision

RUN pip install imageio
RUN pip install tensorboard_logger

WORKDIR "/home"

RUN git clone https://github.com/tabzraz/RL.git

CMD ["/bin/bash"]
