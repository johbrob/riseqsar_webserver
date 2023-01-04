FROM condaforge/mambaforge:4.14.0-0 as conda
FROM nvidia/cuda:11.3.0-base as base

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

COPY --from=conda /opt/conda /opt/conda

# add conda to PATH
ENV CONDA_DIR /opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

WORKDIR /app

COPY . .

RUN conda create --copy -p /env --file conda-linux-64.lock && \
    conda run -p /env python -m pip install --no-deps rdkit==2022.03.5 && \
    conda run -p /env python -m pip install --no-deps ./ChEMBL_Structure_Pipeline && \
    conda clean -afy

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
