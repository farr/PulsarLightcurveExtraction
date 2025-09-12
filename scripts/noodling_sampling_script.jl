using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

# using Distributed

Nmcmc = 1000
Nchain = 4

# addprocs(Nchain)

# @everywhere begin 
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))

    using ArviZ
    using Distributions
    using FITSIO
    using MCMCChainsStorage
    using Mooncake
    using NCDatasets
    using PulsarLightcurveExtraction
    using Random
    using StatsBase
    using Turing
# end

f = FITS(joinpath(@__DIR__, "..", "data", "J0030+0451_merged_phase_0.25-3keV.fits"))
times = read(f[2], "TIME")
phases = read(f[2], "PULSE_PHASE")
segment_starts = read(f[3], "START")
segment_ends = read(f[3], "STOP")

n_fourier = 10
si = segment_indices(times, segment_starts, segment_ends)
cm, sm = cos_sin_matrices(phases,n_fourier)

nseg = length(segment_ends)
sel = times .< segment_ends[nseg]

T = sum(segment_ends[1:nseg] .- segment_starts[1:nseg])
mean_rate = sum(sel)/T

ph = 0.0:0.01:1.0
lc_cos_matrix, lc_sin_matrix = cos_sin_matrices(ph, n_fourier)

model = varying_background_fourier_model(si[sel], segment_starts[1:nseg], segment_ends[1:nseg], cm[sel,:], sm[sel,:], lc_cos_matrix, lc_sin_matrix)

# For optimization
avg_rate = length(si) / T

mu_log_bg = log(avg_rate)
sigma_log_bg = 1.0
log_bg_segment = log(mean_rate) * ones(nseg)
sigma_beta_scaled = 1.0
beta_cos = zeros(n_fourier)
beta_sin = zeros(n_fourier)

map_point = maximum_a_posteriori(model; initial_params=[mu_log_bg; sigma_log_bg; log_bg_segment; sigma_beta_scaled; beta_cos; beta_sin], adtype=AutoMooncake(; config=nothing))

# Start sampling at the optimum
trace = sample(model, NUTS(Nmcmc, 0.85; adtype=AutoMooncake(; config=nothing)), initial_params=map_point.values.array) # MCMCDistributed(), Nmcmc, Nchain)
genq = generated_quantities(model, trace)
trace = append_generated_quantities(trace, genq)
chain = from_mcmcchains(trace; 
                        coords=Dict(:segment=>1:nseg, 
                                    :fourier=>1:n_fourier,
                                    :phase=>ph),
                        dims=Dict(:log_bg_segment_scaled=>(:segment,),
                                  :bg_segment=>(:segment,),
                                  :beta_cos_scaled=>(:fourier,),
                                  :beta_sin_scaled=>(:fourier,),
                                  :beta_cos=>(:fourier,),
                                  :beta_sin=>(:fourier,),
                                  :lc=>(:phase,)))

to_netcdf(chain, joinpath(@__DIR__, "..", "data", "J0030+0451_varying_background_fourier.nc"))
