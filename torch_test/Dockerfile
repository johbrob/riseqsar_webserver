# syntax=docker/dockerfile:

#FROM 1.11.0-cuda11.3-cudnn8-runtime
FROM pytorch/pytorch

#COPY requirements.txt requirements.txt
#RUN pip3 install -r requirements.txt

COPY torch_environment.yml torch_environment.yml
RUN conda env create -f torch_environment.yml


COPY . .

#CMD ["conda", "activate", "rise-qsar-torch"]
#CMD ["python3", "api.py"]
CMD ["/bin/bash"]
