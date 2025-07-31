#!/usr/bin/env bash

# This script generates two datasets for spatial and temporal scaling
# Both datasets will be dumped
SPATIAL_ROOT="/tmp/spatial" # Path to spatial scaling datasets
TEMPORAL_ROOT="/tmp/temporal" # Path to temporal scalingdatasets

# Generate spatial scaling dataset
SCALES="10 20 40 80"
SPATIAL_N=500 # 4 x 500 = 2000
for s in $SCALES; do
    python datasets/datasets/main.py ${SPATIAL_N} ${SPATIAL_ROOT} \
        --translation --scaling $s
done

# Generate temporal scaling dataset
VELOCITIES="0.16 0.32 0.64 1.28"
TEMPORAL_N=2000
python datasets/datasets/main.py ${TEMPORAL_N} ${TEMPORAL_ROOT} \
    --max_velocities ${VELOCITIES} --scaling
