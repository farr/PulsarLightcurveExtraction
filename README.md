# Extracting NICER Lightcurves with Variable Background

This repo is a complete mess, as I play around with ideas for extracting pulsar
lightcurves from NICER data using a background model that varies by time.

## Running scripts

Note: to run the scripts in `scripts/`, you will need to activate the local env, as in 

```bash
julia --project=. scripts/J0740_sampling_script.jl
```

or else run from a development environment with the local package added, as in

```julia
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
include("scripts/J0740_sampling_script.jl")
```

(this will happen automatically in VSCode if you open the folder as a project
and run the script from there).