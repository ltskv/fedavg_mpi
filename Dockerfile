from ubuntu:18.04

RUN apt -y update && \
        apt -y install build-essential pkg-config ninja-build python3 \
        python3-pip python3-dev mpich && \
        apt clean

RUN pip3 install --no-cache \
        meson==0.51.2 \
        numpy==1.16.4 \
        tensorflow==1.14.0 \
        Flask==1.0.2 \
        cython==0.29.14
RUN mkdir /workspace
COPY bridge.pyx library.py server.py meson.build main.c /workspace/
RUN cd /workspace && meson build && cd build && ninja
WORKDIR /workspace
