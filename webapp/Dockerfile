#syntax=docker/dockerfile:1

FROM condaforge/mambaforge:4.14.0-0 as conda

EXPOSE 5000
WORKDIR /app

RUN apt-get --assume-yes update && \
    apt-get --assume-yes upgrade && \
    apt-get --assume-yes install libxrender1 && \
    apt-get --assume-yes install libxtst6 && \
    rm -rf /var/lib/apt/lists/*

COPY context/ environment/conda-linux-64.lock .

RUN mamba create --copy -p /env --file conda-linux-64.lock && \
    conda clean -afy && \
    mamba run -p /env python -m pip install python-dotenv==0.21.0 && \
    mamba run -p /env python -m pip install --no-deps rdkit==2022.03.5

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1   # prevent Python from making pyc files to disk
ENV PYTHONUNBUFFERED 1          # prevents Python from buffering stdout and stderr

CMD [ "/env/bin/gunicorn", "-b", "0.0.0.0:5000", "--worker-tmp-dir", "/dev/shm", "--workers", "2", \
    "--threads", "4", "--worker-class", "gthread", "run:app"]