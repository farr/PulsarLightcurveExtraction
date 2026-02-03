## Initialize the env
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

## Set up script parameters
n_spec = 50
n_segments = nothing # Do all segments
n_fourier = 10

fg_scale = 5e-6 # Empirically determined fg rate estimate, based on not constraining the posterior too much.

n_chain = 4
n_mcmc = 1000

## Set up distributed sampling
using Distributed
addprocs(n_chain)

## Load packages everywhere
@everywhere begin
    using ArviZ
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
m = cat(cm, sm, dims=2)

event_segment_indices = PulsarLightcurveExtraction.segment_indices(event_time, segment_start, segment_stop);
event_areas = PulsarLightcurveExtraction.event_areas(event_time, event_pi, arf_start, arf_stop, arf_e_low, arf_e_high, arf_response);
event_spectral_indices, spec_bins_pi = PulsarLightcurveExtraction.spectral_indices(event_pi, n_spec)

spec_bin_centers = 0.5 .* (spec_bins_pi[1:end-1] .+ spec_bins_pi[2:end]) .* PulsarLightcurveExtraction.PI_TO_KEV

# Scale design matrix by event areas, so that the fg_coeffs are in units of counts per square cm per second.
m = m .* reshape(event_areas, (:, 1))

## Cut down the samples, if necessary
if n_segments !== nothing
    event_sel = event_segment_indices .<= n_segments

    event_time = event_time[event_sel]
    event_segment_indices = event_segment_indices[event_sel]
    event_areas = event_areas[event_sel]
    event_spectral_indices = event_spectral_indices[event_sel]
    m = m[event_sel, :]

    segment_start = segment_start[1:n_segments]
    segment_stop = segment_stop[1:n_segments]
    segment_Ts = segment_Ts[1:n_segments]
end

## Estimate helper quantities from the background
est_log_bg, est_log_bg_uncert = PulsarLightcurveExtraction.estimate_log_bg(event_time, segment_start, segment_stop)

mean_est_log_bg = mean(est_log_bg)
std_est_log_bg = std(est_log_bg)

est_bg_spec, est_bg_spec_uncert = PulsarLightcurveExtraction.estimate_bg_spec(event_spectral_indices)

## Set up the model
model = PulsarLightcurveExtraction.spec_fourier_model(m, event_segment_indices, event_spectral_indices, segment_Ts, est_log_bg, est_log_bg_uncert, est_bg_spec, est_bg_spec_uncert, fg_scale)

## Sample it
chains = sample(model, NUTS(n_mcmc, 0.65; adtype=AutoMooncake()), MCMCDistributed(), n_mcmc, n_chain)

## Package it up
trace = from_mcmcchains(chains; dims=Dict(:fg_coeffs => (:fourier,), :log_dbg_segment => (:segment,), :log_bg_segment => (:segment,), :bg_segment => (:segment,), :bg_spec => (:energy,), :fg_spec => (:energy,)), coords=Dict(:fourier => 1:size(m,2), :segment => 1:n_segments, :energy => spec_bin_centers))

## Save the chains
to_netcdf(trace, joinpath(@__DIR__, "..", "data", "J0740_trace.nc"))