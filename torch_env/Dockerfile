# syntax=docker/dockerfile:1

FROM pytorch/pytorch
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

CMD ["python3", "api.py"]