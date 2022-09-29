# syntax=docker/dockerfile:1

FROM nvidia/cuda:9.0-cudnn7-devel
#FROM nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04
FROM python:3.9-slim-buster

EXPOSE 5000

WORKDIR /app

COPY ChEMBL_Structure_Pipeline ChEMBL_Structure_Pipeline
COPY requirements.txt .
COPY api.py predictor_loading.py setup.py ./
COPY riseqsar riseqsar
COPY experiments experiments

RUN python -m pip install --upgrade pip && \
 	pip install -r requirements.txt && \
	pip install ./ChEMBL_Structure_Pipeline && \
	pip install -e .

RUN apt-get update && \
	apt-get -y install protobuf-compiler cmake git
RUN apt-get install build-essential -y
RUN	git clone https://github.com/Xtra-Computing/thundersvm.git
#RUN	cd thundersvm
RUN	mkdir thundersvm/build
#RUN cd build
#RUN ls
#RUN ls thundersvm
#RUN ls thundersvm/build
#RUN ls ..
#RUN ls ../usr
#RUN whereis gcc
#RUN whereis g++
#RUN LIST(APPEND CMAKE_PROGRAM_PATH  "c:/usr/bin/" ...)
RUN apt-get install nvidia-cuda-toolkit -y
RUN find / -type d -name 'cuda'
WORKDIR /app/thundersvm/build
RUN whereis cudatoolkit
RUN ls /usr/local
#RUN ls /usr/local/cuda
#RUN ln -s /usr/local/cuda-5.5 /usr/local/cuda
RUN cmake ..
RUN make -j



CMD ["python3", "api.py"]