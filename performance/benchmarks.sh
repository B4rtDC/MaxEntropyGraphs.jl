#bin/bash

## run it standalone:
## Note: this script will take a very long time to run (>24Hrs), 
## so it is recommended to run it in the background.
## nohup ./benchmarks.sh >> benchmark.log 2>&1 &

## Generate conda environment for benchmarking
conda env create -f benchmarking.yml
conda activate benchmarking

## Generate Julia environment for benchmarking
julia --project=. -e 'using Pkg; Pkg.instantiate()'

## Run benchmarks (Julia will generate the necessary shell and Python scripts)
echo "$(date) - Starting Julia benchmarks"
julia --project=. -t auto ./UBCM_benchmarks.jl
julia --project=. -t auto ./BiCM_benchmarks.jl
echo "$(date) - Finished Julia benchmarks"

## Run Python benchmark scripts
echo "$(date) - Starting Python benchmarks"
sh ./UBCM_script.sh
sh ./BiCM_script.sh
echo "$(date) - Finished Python benchmarks"

## Generate the plots
echo "$(date) - Generating the plots"
julia --project=. -e './UBCM_plots.jl'
julia --project=. -e './BiCM_plots.jl'


## Deactivate conda environment
conda deactivate

echo "$(date) - Finished benchmarks"