#!/usr/bin/env bash

SING=tensorflow.sif

if [ -f "$SING" ]; then
    echo "Using $SING"
else
    srun --cpus-per-task=8 --pty singularity build $SING docker://nvcr.io/nvidia/tensorflow:22.03-tf1-py3
fi

cp /user/share/scripts/jupscript.sh jupscript.sh
mkdir -p $HOME/runuser/
srun --qos=short --cpus-per-task=8 --pty --gres=gpu:1 --mem=128000M singularity exec -B $HOME/runuser/:/run/user/$(id -u) --nv $SING ./jupscript.sh
#srun --qos=short --cpus-per-task=8 --pty --gres=gpu:4 --mem=128000M singularity exec -B $HOME/runuser/:/run/user/$(id -u) --nv $SING ./jupscript.sh
rm jupscript.sh
