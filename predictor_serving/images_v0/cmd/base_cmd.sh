#!/bin/bash
#source ~/anaconda3/etc/profile.d/conda.sh
. /opt/conda/etc/profile.d/conda.sh || ~/../opt/conda/etc/profile.d/conda.sh
conda activate base-predictor
gunicorn -b 0.0.0.0:5000 --worker-tmp-dir /dev/shm --workers 2 --threads 4 --worker-class gthread api:app
