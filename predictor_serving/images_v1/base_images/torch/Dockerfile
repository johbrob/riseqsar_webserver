FROM nvidia_cuda113_conda

EXPOSE 5000
WORKDIR /app

RUN conda run -p /env python -m pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1


