FROM nvidia_cuda113_pytorch1.12 

RUN conda run -p /env python -m pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.12.0+cu113.html && \
    conda run -p /env python -m pip install torch-geometric==2.0
