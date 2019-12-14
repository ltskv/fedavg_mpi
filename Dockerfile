from ubuntu:18.04

RUN apt -y update && \
        apt -y install build-essential pkg-config ninja-build python3 python3-pip python3-dev mpich
RUN pip3 install meson numpy tensorflow flask cython
RUN mkdir /workspace
COPY bridge.pyx library.py server.py meson.build main.c /workspace/
RUN cd /workspace && meson build && cd build && ninja
WORKDIR /workspace
