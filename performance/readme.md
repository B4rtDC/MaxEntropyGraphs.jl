# Benchmarking
This is the benchmarking folder that allows users to replicate the benchmarks and performance claims related to `MaxEntropyGraphs.jl`.

## Setup 
To create the appropriate python environment, the `benchmarking.yml` file is provide. The python environment can be created as follows:
```bash
conda env create -f ./benchmarking.yml
```

The appropriate Julia environment is available through the `Project.toml` file in this folder.

*Notes*:
- The benchmark currently refers to the github of the package due to the package registry not being at the latest release;
- `julia` is assumed to be a known command;
- conda is assumed to be available for the user.

## Principle
The relevant common functions are defined in `benchmark_helpers.jl`. For each model, a separate script exists:
* UBCM
* DBCM
* BiCM

These scripts work as follows: 
1. The Julia script defines the graphs and writes them out to edge lists so that the same graphs will be used in Python;
2. The Julia script also creates the associate python scripts for benchmarking NEMtropy and will run them;
3. The Python script(s) will be run.

The `data` subfolder holds the graph information (as edge lists). 
Sampled graphs are stored in the `sample` subfolder. 
Raw benchmark data is stored in the `benchmarks` subfolder, in an appropriate subfolder to match the specific Python and Julia version.
A logging file is used to keep track of the progress.  
The plot script will generate the visualizations, which will be stored in the `plots` subfolder. 