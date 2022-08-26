# syntax=docker/dockerfile:

#FROM 1.11.0-cuda11.3-cudnn8-runtime
FROM pytorch/pytorch
FROM python:3.9-slim-buster

#CMD apt install python3.9
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

#COPY torch_environment.yml torch_environment.yml
#RUN conda env create -f torch_environment.yml


COPY . .
RUN pip install ./ChEMBL_Structure_Pipeline

#CMD ["conda", "activate", "rise-qsar-torch"]
CMD pip install -e .
CMD ["python3", "api.py"]
