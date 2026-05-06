## Parse command-line arguments
using ArgParse

let s = ArgParseSettings(description="Sample the J0740 pulsar lightcurve model.")
    @add_arg_table! s begin
        "--n-spec"
            help = "Number of spectral bins"
            arg_type = Int
            default = 8
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
        "--e-min"
            help = "Minimum energy (keV)"
            arg_type = Float64
            default = 0.2
        "--e-max"
            help = "Maximum energy (keV)"
            arg_type = Float64
            default = 3.1
        "--spec-order"
            help = "Order of the splines in the spectral model (default: 4, i.e. cubic splines, with continuous second derivatives)"
            arg_type = Int
            default = 4
        "--pi-min"
            help = "Minimum PI to use (default: 25, i.e. ~0.25 keV)"
            arg_type = Int
            default = 25
        "--pi-max"
            help = "Maximum PI to use (default: 300, i.e. ~3.0 keV)"
            arg_type = Int
            default = 300
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
        "--use-mooncake"
            help = "Whether to use Mooncake for AD (default: false, i.e. use Enzyme)"
            action = :store_true
        "--fisher-information-ordering"
            help = "Whether to order segments by their Fisher information content (default: false, i.e. use the segments in the order of observation)"
            action = :store_true
    end
    global parsed_args = parse_args(s)
end

## Set up script parameters
n_spec = parsed_args["n-spec"]
n_segments = parsed_args["n-segments"]
n_fourier = parsed_args["n-fourier"]
fractional_variability = parsed_args["fractional-variability"]
e_min = parsed_args["e-min"]
e_max = parsed_args["e-max"]
spec_order = parsed_args["spec-order"]
pi_min = parsed_args["pi-min"]
pi_max = parsed_args["pi-max"]
n_chain = parsed_args["n-chain"]
n_mcmc = parsed_args["n-mcmc"]
target_arate = parsed_args["target-arate"]
use_mooncake = parsed_args["use-mooncake"]
fisher_information_ordering = parsed_args["fisher-information-ordering"]

trace_suffix = (n_segments === nothing ? "" : "_$(n_segments)")
outpath = joinpath(@__DIR__, "..", "data", "J0740_trace$(trace_suffix).nc")

using Distributed
if n_chain > 1
    @info "Adding $n_chain processes for MCMC sampling..."
    addprocs(n_chain)
end

## Load packages
@everywhere begin
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

    # Periodically flush stdout/stderr so @info, @progress, etc. appear promptly
    # when running non-interactively under SLURM.
    const _flush_timer = Timer(0; interval=1.0) do _
        flush(stdout)
        flush(stderr)
    end

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

arf_e_high, arf_e_low, arf_e, arf_response, arf_start, arf_stop = h5open(joinpath(@__DIR__, "..", "data", "J0740_merged_arf.h5"), "r") do f
    map(x -> read(f, x), ("ENERG_HI", "ENERG_LO", "ENERGY", "SPECRESP", "START", "STOP"))
end

rmf_e_high, rmf_e_low, rmf_response, rmf_start, rmf_stop = h5open(joinpath(@__DIR__, "..", "data", "J0740_merged_rmf.h5"), "r") do f
    rmf_e_high = read(f, "ENERG_HI")
    rmf_e_low = read(f, "ENERG_LO")
    rmf_response = read(f, "MATRIX")
    rmf_start = read(f, "START")
    rmf_stop = read(f, "STOP")

    return rmf_e_high, rmf_e_low, rmf_response, rmf_start, rmf_stop
end

## Check consistency with min_pi and max_pi:
@assert (all(pi_min .<= event_pi) && all(event_pi .<= pi_max)) "Event PIs must be between pi_min and pi_max."

## Construct design matrix, bin indices, etc, etc
cm, sm = PulsarLightcurveExtraction.cos_sin_matrices(event_phase, n_fourier)

event_segment_indices = PulsarLightcurveExtraction.segment_indices(event_time, segment_start, segment_stop);

fg_spline_to_pi, bg_spline_to_pi = PulsarLightcurveExtraction.construct_spline_spectral_bases(e_min, e_max, n_spec, spec_order, arf_e_low, arf_e_high, arf_response, rmf_response)
fg_spectral_design_matrix, bg_spectral_design_matrix = PulsarLightcurveExtraction.spectral_design_matrices(event_time, event_pi, fg_spline_to_pi, bg_spline_to_pi, rmf_start, rmf_stop)

## Cut down the samples, if necessary
if n_segments !== nothing
    if fisher_information_ordering
        est_fi = segment_fisher_estimate(event_segment_indices, segment_start, segment_stop)
        analysis_segment_inds = sortperm(est_fi)[end:-1:end-n_segments+1] # Start at the most informative segment, and proceed down the list.
    else
        analysis_segment_inds = collect(1:n_segments)
    end

    analysis_segment_inds_set = Set(analysis_segment_inds)
    event_sel = [esi in analysis_segment_inds_set for esi in event_segment_indices]

    event_time = event_time[event_sel]

    event_segment_indices = event_segment_indices[event_sel]

    # Now the indices may be messed up; they need to be re-mapped so that index X becomes I where I is the index in the sorted array in which it appears
    esi_map = Dict(si => i for (i, si) in enumerate(analysis_segment_inds))
    event_segment_indices = [esi_map[esi] for esi in event_segment_indices]

    fg_spectral_design_matrix = fg_spectral_design_matrix[event_sel, :]
    bg_spectral_design_matrix = bg_spectral_design_matrix[event_sel, :]

    cm = cm[event_sel, :]
    sm = sm[event_sel, :]

    segment_start = segment_start[analysis_segment_inds]
    segment_stop = segment_stop[analysis_segment_inds]
else
    n_segments = length(segment_start)
    analysis_segment_inds = collect(1:n_segments)
end

## Total exposure over all segments in use
fg_exposure, bg_exposure = PulsarLightcurveExtraction.foreground_background_exposure(pi_min, pi_max, segment_start, segment_stop, arf_start, arf_stop, fg_spline_to_pi, bg_spline_to_pi)

## Set up the model
model = PulsarLightcurveExtraction.spec_fourier_model(cm, sm, fg_spectral_design_matrix, bg_spectral_design_matrix, event_segment_indices, fg_exposure, bg_exposure, fractional_variability)

## Set up the autodiff
if use_mooncake
    @info "Using Mooncake for AD"
    adtype = AutoMooncake()
else
    @info "Using Enzyme for AD"
    adtype = AutoEnzyme(mode=Enzyme.set_runtime_activity(Enzyme.Reverse))
end

## Initialization
@info "Initializing from U(-2,2) in unconstrained space"
if n_chain == 1
    init_params = InitFromUniform()
else
    init_params = [InitFromUniform() for _ in 1:n_chain]
end

## Kernel
kernel = NUTS(n_mcmc, target_arate; adtype=adtype)

## Sample it
@info "Sampling with $n_segments segments of data with $n_chain chains using distributed processes..."
if n_chain == 1
    chains = sample(model, kernel, n_mcmc; initial_params=init_params)
else
    chains = sample(model, kernel, MCMCDistributed(), n_mcmc, n_chain; initial_params=init_params)
end

## Package it up
trace = from_mcmcchains(chains;
    dims=Dict(
        :mu_log_bg => (:spec,),
        :mu_bg => (:spec,),
        :sigma_log_bg => (:spec,),
        :log_fg_coeff_const => (:spec,),
        :fg_coeff_const => (:spec,),
        :log_bg => (:spec, :segment),
        :bg => (:spec, :segment),
        :dfg_coeffs_cos => (:spec, :fourier),
        :dfg_coeffs_sin => (:spec, :fourier),
        :fg_coeffs_cos => (:spec, :fourier),
        :fg_coeffs_sin => (:spec, :fourier)),
    coords=Dict(
        :fourier => 1:n_fourier,
        :segment => analysis_segment_inds,
        :spec => 1:n_spec))

## Check minimum ESS:
println("Minimum ESS: ", minimum(ess(trace)))

## Save the chains
to_netcdf(trace, outpath)