#!/bin/bash

eval "$(conda shell.bash hook)"

conda activate Ising

nohup python IsingReduction.py > out.log &
