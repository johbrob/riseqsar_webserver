#syntax=docker/dockerfile:1

FROM condaforge/mambaforge:4.14.0-0 as conda

EXPOSE 5000
WORKDIR /app

#COPY images_v0/environments/conda-linux-64.lock .
#RUN mamba create --copy -p /env --file conda-linux-64.lock && conda clean -afy
#RUN --mount=type=cache,target=/opt/conda/pkgs mamba create --copy -p /env --file conda-linux-64.lock

RUN apt-get --assume-yes update && \
    apt-get --assume-yes upgrade && \
    apt-get --assume-yes install libxrender1
RUN apt-get --assume-yes install libxtst6 
    #, libxext6, libSM, libXrender

COPY . .

RUN mamba env create && \
    mamba  run -n riseqsar-webserver python -m pip install python-dotenv==0.21.0 && \
    mamba run -n riseqsar-webserver python -m pip install --no-deps rdkit

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1   # prevent Python from making pyc files to disk
ENV PYTHONUNBUFFERED 1          # prevents Python from buffering stdout and stderr


#CMD ["mamba", "run", "-n", "base-predictor", "gunicorn", "-w", "4", "-b", "localhost:5000", "api:app"]
CMD ["sh", "cmd.sh"]
