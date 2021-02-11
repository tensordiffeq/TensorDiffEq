FROM tensorflow/tensorflow:nightly-gpu

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y eog python3-tk python-yaml texlive-full openssh-server sudo x11-apps && apt-get clean && rm -rf /var/lib/apt/lists

ENV DISPLAY=:0

RUN pip install librosa pytz matplotlib scikit-learn Pillow pandas progress openpyxl numpy pyDOE numba tensordiffeq

RUN useradd -rm -d /home/ubuntu -s /bin/bash -g root -G sudo -u 1000 test

RUN echo 'test:test' | chpasswd

RUN service ssh start

EXPOSE 22

ENV QT_X11_NO_MITSHM=1
