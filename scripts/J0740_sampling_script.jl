## Set up script parameters
n_spec = 16
n_segments = nothing
n_fourier = 4

fg_scale = 1e-6 # Empirically determined fg rate estimate, based on not constraining the posterior too much.

n_chain = 8
n_mcmc = 1000

target_arate = 0.8

## Set up distributed sampling
using Distributed
if n_chain > 1
    addprocs(n_chain)
end

## Load packages everywhere
@everywhere begin
    using ArviZ
    using DimensionalData
    using FITSIO
    using HDF5
    using LinearAlgebra
    using Mooncake
    using NCDatasets
    using PulsarLightcurveExtraction
    using Turing

    # Otherwise the sampler will try to use multiple threads for linear algebra, alas!
    BLAS.set_num_threads(1)
end

## Load data
event_time, event_phase, event_pi, segment_start, segment_stop = FITS(joinpath(@__DIR__, "..", "data", "J0740_merged_phase_0.25-3keV.fits.gz"), "r") do f
    event_time = read(f[2], "TIME")
    event_phase = read(f[2], "PULSE_PHASE")
    event_pi = read(f[2], "PI")

    segment_start = read(f[4], "START")
    segment_stop = read(f[4], "STOP")

    (event_time, event_phase, event_pi, segment_start, segment_stop)
end

segment_Ts = segment_stop .- segment_start

arf_e_high, arf_e_low, arf_e, arf_response, arf_start, arf_stop = h5open(joinpath(@__DIR__, "..", "data", "J0740_merged_arf.h5"), "r") do f
    map(x -> read(f, x), ("ENERG_HI", "ENERG_LO", "ENERGY", "SPECRESP", "START", "STOP"))
end

## Construct design matrix, bin indices, etc, etc
cm, sm = PulsarLightcurveExtraction.cos_sin_matrices(event_phase, n_fourier)

event_segment_indices = PulsarLightcurveExtraction.segment_indices(event_time, segment_start, segment_stop);
event_spectral_indices, spec_bins_pi = PulsarLightcurveExtraction.spectral_indices(event_pi, n_spec)
event_areas = PulsarLightcurveExtraction.event_areas(event_time, event_spectral_indices, spec_bins_pi, arf_start, arf_stop, arf_e_low, arf_e_high, arf_response);

# Logarithmic bins assumed!
spec_bin_centers = sqrt.(spec_bins_pi[1:end-1] .* spec_bins_pi[2:end]) .* PulsarLightcurveExtraction.PI_TO_KEV

## Cut down the samples, if necessary
if n_segments !== nothing
    event_sel = event_segment_indices .<= n_segments

    event_time = event_time[event_sel]
    event_segment_indices = event_segment_indices[event_sel]
    event_areas = event_areas[event_sel]
    event_spectral_indices = event_spectral_indices[event_sel]

    cm = cm[event_sel, :]
    sm = sm[event_sel, :]

    segment_start = segment_start[1:n_segments]
    segment_stop = segment_stop[1:n_segments]
    segment_Ts = segment_Ts[1:n_segments]
else
    n_segments = length(segment_start)
end

## Total exposure over all segments in use
energy_bin_exposures = PulsarLightcurveExtraction.energy_bin_exposure(spec_bins_pi, segment_start, segment_stop, arf_start, arf_stop, arf_e_low, arf_e_high, arf_response)

## Estimate helper quantities from the background
est_log_bg, est_log_bg_uncert = PulsarLightcurveExtraction.estimate_log_bg(event_segment_indices, event_spectral_indices, segment_start, segment_stop)
est_log_fg_const, est_log_fg_const_uncert = PulsarLightcurveExtraction.estimate_log_fg_const(event_spectral_indices, energy_bin_exposures)

## Set up the model
model = PulsarLightcurveExtraction.spec_fourier_model(cm, sm, event_segment_indices, event_spectral_indices, event_areas, segment_Ts, energy_bin_exposures, est_log_bg, est_log_bg_uncert, est_log_fg_const, est_log_fg_const_uncert, fg_scale)

if n_chain > 1
    println("Running with $n_chain chains in distributed mode...")
else
    println("Running with a single chain.")
end

## Sample it
if n_chain > 1
    chains = sample(model, NUTS(n_mcmc, target_arate; adtype=AutoMooncake()), MCMCDistributed(), n_mcmc, n_chain)
else
    chains = sample(model, NUTS(n_mcmc, target_arate; adtype=AutoMooncake()), n_mcmc)
end

## Package it up
trace = from_mcmcchains(chains; dims=Dict(:dmu_log_bg => (:energy, ), :mu_log_bg => (:energy, ), :log_dsigma_log_bg => (:energy, ), :sigma_log_bg => (:energy,), :dlog_bg => (:energy, :segment), :log_bg => (:energy, :segment), :bg => (:energy, :segment), :dlog_fg_coeff_const => (:energy,), :log_fg_coeff_const => (:energy,), :fg_coeff_const => (:energy,), :dfg_coeffs_cos => (:energy, :fourier), :dfg_coeffs_sin => (:energy, :fourier), :fg_coeffs_cos => (:energy, :fourier), :fg_coeffs_sin => (:energy, :fourier)), coords=Dict(:fourier => 1:n_fourier, :segment => 1:n_segments, :energy => spec_bin_centers))

## Save the chains
to_netcdf(trace, joinpath(@__DIR__, "..", "data", "J0740_trace.nc"))
