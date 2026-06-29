#!/bin/sh
# Submit a GPUMD job from each immediate subdirectory.
set -eu

for dir in */; do
    [ -d "$dir" ] || continue
    echo ">>> Entering $dir and submitting gpumd.slurm"
    cd "$dir"
    sbatch gpumd.slurm
    cd ..
    wait
done

echo "All subdirectory jobs have been submitted."
