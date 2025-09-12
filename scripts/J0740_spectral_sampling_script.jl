using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Distributed

Nmcmc = 1000
Nchain = 4

addprocs(Nchain)

@everywhere begin 
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))

    using ArviZ
    using CategoricalArrays
    using Distributions
    using FITSIO
    using MCMCChainsStorage
    using Mooncake
    using NCDatasets
    using PulsarLightcurveExtraction
    using Random
    using StatsBase
    using Turing
end

f = FITS(joinpath(@__DIR__, "..", "data", "J0740_merged_phase_0.25-3keV.fits.gz"))
times = read(f[2], "TIME")
phases = read(f[2], "PULSE_PHASE")
pi_channels = read(f[2], "PI")
segment_starts = read(f[4], "START")
segment_ends = read(f[4], "STOP")

pi_channels_categorical = CategoricalArray(pi_channels)
pi_indices = levelcode.(pi_channels_categorical)
nchannels = length(levels(pi_channels_categorical))

n_fourier = 10
si = segment_indices(times, segment_starts, segment_ends)
cm, sm = cos_sin_matrices(phases,n_fourier)

nseg = 10 # length(segment_ends)
sel = times .< segment_ends[nseg]

T = sum(segment_ends[1:nseg] .- segment_starts[1:nseg])
mean_rate = sum(sel)/T

ph = 0.0:0.01:1.0
lc_cos_matrix, lc_sin_matrix = cos_sin_matrices(ph, n_fourier)

model = varying_background_spectral_fourier_model(pi_indices[sel], si[sel], segment_starts[1:nseg], segment_ends[1:nseg], cm[sel,:], sm[sel,:], lc_cos_matrix, lc_sin_matrix)

# For optimization
avg_rate = length(si) / T

mu_log_bg = log(avg_rate)
sigma_log_bg = 1.0
log_bg_segment = log(mean_rate) * ones(nseg)
sigma_beta_scaled = 1.0
beta_cos = zeros(n_fourier)
beta_sin = zeros(n_fourier)
bg_spec = ones(nchannels) ./ nchannels
fg_spec = ones(nchannels, n_fourier) ./ nchannels

# map_point = maximum_a_posteriori(model; initial_params=[mu_log_bg; sigma_log_bg; log_bg_segment; sigma_beta_scaled; beta_cos; beta_sin], adtype=AutoMooncake(; config=nothing))
initial_params = [mu_log_bg; sigma_log_bg; log_bg_segment; sigma_beta_scaled; beta_cos; beta_sin; bg_spec; vec(fg_spec)]
initial_params = [initial_params for i in 1:Nchain]

# Start sampling at a point where the model won't have negative intensity
trace = sample(model, NUTS(Nmcmc, 0.85; adtype=AutoMooncake(; config=nothing)), MCMCDistributed(), Nmcmc, Nchain; initial_params=initial_params)
genq = generated_quantities(model, trace)
trace = append_generated_quantities(trace, genq)
chain = from_mcmcchains(trace; 
                        coords=Dict(:segment=>1:nseg, 
                                    :fourier=>1:n_fourier,
                                    :phase=>ph,
                                    :channel=>levels(pi_channels_categorical)),
                        dims=Dict(:log_bg_segment_scaled=>(:segment,),
                                  :bg_segment=>(:segment,),
                                  :beta_cos_scaled=>(:fourier,),
                                  :beta_sin_scaled=>(:fourier,),
                                  :beta_cos=>(:fourier,),
                                  :beta_sin=>(:fourier,),
                                  :bg_spec=>(:channel,),
                                  :fg_spec=>(:channel, :fourier),
                                  :lc=>(:phase,)))

to_netcdf(chain, joinpath(@__DIR__, "..", "data", "J0740_spectral_varying_background_fourier.nc"))
