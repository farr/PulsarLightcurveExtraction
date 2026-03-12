## Parse command-line arguments
using ArgParse

let s = ArgParseSettings(description="Sample the J0740 pulsar lightcurve model.")
    @add_arg_table! s begin
        "--n-spec"
            help = "Number of spectral bins"
            arg_type = Int
            default = 16
        "--n-segments"
            help = "Number of segments to use (default: all)"
            arg_type = Int
            default = nothing
        "--n-fourier"
            help = "Number of Fourier terms"
            arg_type = Int
            default = 4
        "--fractional-variability"
            help = "Fractional variability in the lightcurve"
            arg_type = Float64
            default = 0.1
        "--n-chain"
            help = "Number of MCMC chains"
            arg_type = Int
            default = 4
        "--n-mcmc"
            help = "Number of MCMC samples per chain"
            arg_type = Int
            default = 1000
        "--target-arate"
            help = "Target acceptance rate for NUTS"
            arg_type = Float64
            default = 0.8
    end
    global parsed_args = parse_args(s)
end

## Set up script parameters
n_spec = parsed_args["n-spec"]
n_segments = parsed_args["n-segments"]
n_fourier = parsed_args["n-fourier"]
fractional_variability = parsed_args["fractional-variability"]
n_chain = parsed_args["n-chain"]
n_mcmc = parsed_args["n-mcmc"]
target_arate = parsed_args["target-arate"]

trace_suffix = (n_segments === nothing ? "" : "_$(n_segments)")

## Load packages
using ArviZ
using DimensionalData
using FITSIO
using Enzyme
using HDF5
using LinearAlgebra
using Mooncake
using NCDatasets
using PulsarLightcurveExtraction
using Turing

# Otherwise the sampler will try to use multiple threads for linear algebra, alas!
BLAS.set_num_threads(1)

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

# Logarithmic bins assumed!
spec_bin_centers = sqrt.(spec_bins_pi[1:end-1] .* spec_bins_pi[2:end]) .* PulsarLightcurveExtraction.PI_TO_KEV

## Cut down the samples, if necessary
if n_segments !== nothing
    event_sel = event_segment_indices .<= n_segments

    event_time = event_time[event_sel]
    event_segment_indices = event_segment_indices[event_sel]
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
energy_bin_areas = PulsarLightcurveExtraction.energy_bin_areas(spec_bins_pi, segment_start, segment_stop, arf_start, arf_stop, arf_e_low, arf_e_high, arf_response)

## Set up the model
model = PulsarLightcurveExtraction.spec_fourier_model(cm, sm, event_segment_indices, event_spectral_indices, segment_Ts, energy_bin_areas; fractional_variability=fractional_variability)

println("Running with $n_chain chains using $(Threads.nthreads()) threads...")

## Sample it
chains = sample(model, NUTS(n_mcmc, target_arate; adtype=AutoEnzyme(mode=Enzyme.set_runtime_activity(Enzyme.Reverse))), MCMCThreads(), n_mcmc, n_chain; callback=flush_stderr_stdout_callback)

## Package it up
trace = from_mcmcchains(chains; dims=Dict(:mu_log_bg => (:energy, ), :sigma_log_bg => (:energy,), Symbol("corr_chol.L") => (:energy, :energy2), :L_cov => (:energy, :energy2), :cov_log_bg => (:energy, :energy2), :log_fg_coeff_const => (:energy,), :fg_coeff_const => (:energy,), :log_bg_uncentered => (:energy, :segment), :log_bg => (:energy, :segment), :bg => (:energy, :segment), :dfg_coeffs_cos => (:energy, :fourier), :dfg_coeffs_sin => (:energy, :fourier), :fg_coeffs_cos => (:energy, :fourier), :fg_coeffs_sin => (:energy, :fourier)), coords=Dict(:fourier => 1:n_fourier, :segment => 1:n_segments, :energy => spec_bin_centers, :energy2 => spec_bin_centers))

## Check minimum ESS:
println("Minimum ESS: ", minimum(ess(trace)))

## Save the chains
to_netcdf(trace, joinpath(@__DIR__, "..", "data", "J0740_trace$(trace_suffix).nc"))
