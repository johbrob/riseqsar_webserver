#!/bin/bash
#source ~/anaconda3/etc/profile.d/conda.sh
. /opt/conda/etc/profile.d/conda.sh || ~/../opt/conda/etc/profile.d/conda.sh
conda activate base
gunicorn -b 0.0.0.0:5000 api:app
