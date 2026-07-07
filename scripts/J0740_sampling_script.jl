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
        "--init-width"
            help = "Width of the uniform initialization distribution in unconstrained space (default: 0.1, i.e. U(-0.1, 0.1))"
            arg_type = Float64
            default = 0.1
        "--use-mooncake"
            help = "Whether to use Mooncake for AD (default: false, i.e. use Enzyme)"
            action = :store_true
        "--fisher-information-ordering"
            help = "Whether to order segments by their Fisher information content (default: false, i.e. use the segments in the order of observation)"
            action = :store_true
        "--init-buffer"
            help = "Number of fast-adaptation steps at the start of warmup (Stan default: 75)"
            arg_type = Int
            default = 75
        "--term-buffer"
            help = "Number of step-size-only adaptation steps at the end of warmup (Stan default: 50)"
            arg_type = Int
            default = 50
        "--window-size"
            help = "Initial mass-matrix adaptation window size (Stan default: 25; doubles each window)"
            arg_type = Int
            default = 25
        "--max-tree-depth"
            help = "Maximum tree depth for NUTS (default: 10; lower values speed up warmup at the cost of some sampling efficiency)"
            arg_type = Int
            default = 10
        "--checkpoint"
            help = "Path to a checkpoint file. If given, sampling state is saved periodically; if the checkpoint(s) already exist, sampling resumes from them instead of Pathfinder initialization (default: no checkpointing)"
            arg_type = String
            default = nothing
        "--checkpoint-every"
            help = "Number of iterations between checkpoint writes (only used if --checkpoint is given)"
            arg_type = Int
            default = 100
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
init_width = parsed_args["init-width"]
use_mooncake = parsed_args["use-mooncake"]
fisher_information_ordering = parsed_args["fisher-information-ordering"]
init_buffer = parsed_args["init-buffer"]
term_buffer = parsed_args["term-buffer"]
window_size = parsed_args["window-size"]
max_tree_depth = parsed_args["max-tree-depth"]
checkpoint_path = parsed_args["checkpoint"]
checkpoint_every = parsed_args["checkpoint-every"]

trace_suffix = (n_segments === nothing ? "" : "_$(n_segments)")
outpath = joinpath(@__DIR__, "..", "data", "J0740_trace$(trace_suffix).nc")

# Set up flushing logger before the first @info so Julia's logging dispatch is
# JIT-compiled in a world age that already knows about FlushingLogger's methods.
using Logging
using TerminalLoggers
using ProgressLogging
using ProgressMeter

struct FlushingLogger <: AbstractLogger
    inner::TerminalLogger
end
Logging.min_enabled_level(l::FlushingLogger) = Logging.min_enabled_level(l.inner)
Logging.shouldlog(l::FlushingLogger, args...) = Logging.shouldlog(l.inner, args...)
Logging.catch_exceptions(l::FlushingLogger) = Logging.catch_exceptions(l.inner)
function Logging.handle_message(l::FlushingLogger, args...; kwargs...)
    Logging.handle_message(l.inner, args...; kwargs...)
    flush(stdout)
    flush(stderr)
end
global_logger(FlushingLogger(TerminalLogger()))

using Distributed
if n_chain > 1
    @info "Adding $n_chain processes for MCMC sampling..."
    addprocs(n_chain)
end

## Load packages
@everywhere begin
    using AbstractMCMC
    using AdvancedHMC
    using ArviZ
    using DimensionalData
    using DynamicPPL
    using FITSIO
    using Enzyme
    using HDF5
    using LinearAlgebra
    using Mooncake
    using NCDatasets
    using PulsarLightcurveExtraction
    using Random
    using Serialization
    using Turing

    # Otherwise the sampler will try to use multiple threads for linear algebra, alas!
    BLAS.set_num_threads(1)
end

# Pathfinder only needs to run on the main process for initialization.
using Pathfinder

## Checkpointing helpers (must be defined on every worker, since sampling of each chain
## may happen on a different process).

# Serialize to a temporary file, then rename, so a process killed mid-write (e.g. by a
# SLURM walltime limit) can never leave a corrupt checkpoint file behind.
@everywhere function atomic_serialize(path, x)
    tmp = path * ".tmp"
    serialize(tmp, x)
    mv(tmp, path; force=true)
end

# Run one chain to `n_total` iterations, checkpointing every `checkpoint_every`
# iterations if `checkpoint_file !== nothing`. If `checkpoint_file` already exists, the
# chain resumes from the saved (position, metric, adaptor, iteration) state rather than
# from `init_params`. `state.state.i` (the sampler's own iteration counter, carried
# inside `state` across calls) is what governs when NUTS adaptation freezes, so a chain
# split across many restarts adapts identically to one run uninterrupted; only draws
# from sampling iterations (state.i > n_adapts) are kept, since warmup draws are
# discarded downstream anyway.
@everywhere function run_chain(model, kernel, n_adapts, n_total, checkpoint_file, checkpoint_every, init_params, progress_channel=nothing)
    rng = Random.default_rng()

    if checkpoint_file !== nothing && isfile(checkpoint_file)
        iter, saved_n_adapts, saved_n_total, state, draws = deserialize(checkpoint_file)
        @assert state.state.i == iter "Checkpoint file $(checkpoint_file) is corrupt: state.state.i ($(state.state.i)) != saved iteration ($(iter))"
        @assert (saved_n_adapts, saved_n_total) == (n_adapts, n_total) "Checkpoint file $(checkpoint_file) was written with n_adapts=$(saved_n_adapts), n_total=$(saved_n_total), but this run requested n_adapts=$(n_adapts), n_total=$(n_total); --n-mcmc (and --checkpoint) must match the original run."
        @info "Resuming chain from checkpoint $(checkpoint_file) at iteration $(iter)/$(n_total)"
        progress_channel !== nothing && put!(progress_channel, iter)
    else
        t, state = AbstractMCMC.step(rng, model, kernel; initial_params=init_params, n_adapts=n_adapts)
        iter = state.state.i
        draws = iter > n_adapts ? [t] : typeof(t)[]
        if checkpoint_file !== nothing
            atomic_serialize(checkpoint_file, (iter, n_adapts, n_total, state, draws))
        end
        progress_channel !== nothing && put!(progress_channel, iter)
    end

    while iter < n_total
        t, state = AbstractMCMC.step(rng, model, kernel, state; n_adapts=n_adapts)
        iter = state.state.i
        if iter > n_adapts
            push!(draws, t)
        end
        progress_channel !== nothing && put!(progress_channel, 1)
        if iter % checkpoint_every == 0 || iter == n_total
            if checkpoint_file !== nothing
                atomic_serialize(checkpoint_file, (iter, n_adapts, n_total, state, draws))
            end
        end
    end

    return draws, state
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

## Find the MLE and Fisher for each segment
const_bg_est = size(bg_spectral_design_matrix, 1) / sum(bg_exposure)
mu_log_bg = log(const_bg_est)
sigma_log_bg = 10.0 # Hard coded large uncertainty just to barely regularize the Fisher.

log_bg_mle = Vector{Float64}[]
log_bg_fisher = Matrix{Float64}[]
@progress "Fisher estimation" for i in axes(bg_exposure, 2)
    sel = event_segment_indices .== i
    mle, fisher_raw = PulsarLightcurveExtraction.segment_bg_mle_and_information(bg_spectral_design_matrix[sel, :], bg_exposure[:, i], mu_log_bg, sigma_log_bg)

    fisher = fisher_raw / (sigma_log_bg*sigma_log_bg) # Fisher is in the *raw* space, want it in the log_bg space

    # Ensure positive-definitness of Fisher.
    fisher = Symmetric(fisher)

    # Check for NaN/Inf before eigvals (which throws an opaque ArgumentError).
    if any(!isfinite, fisher)
        n_nan = count(isnan, fisher)
        n_inf = count(isinf, fisher)
        @warn "Fisher matrix for segment $i (analysis index $(analysis_segment_inds[i])) contains non-finite entries" n_nan n_inf
        @warn "fisher_raw for segment $i" fisher_raw
        @warn "fisher (rescaled) for segment $i" fisher
        @warn "MLE for segment $i" mle
        @warn "bg_exposure[:, $i]" bg_exposure[:, i]
        n_events = count(sel)
        @warn "n_events in segment $i" n_events
    end

    λs = eigvals(fisher)
    λ_min = sqrt(eps(Float64)) * maximum(abs.(λs))
    if minimum(λs) < λ_min
        shift = (λ_min - minimum(λs))
        @info "Shifting FIM for segment $i by $(round(shift, sigdigits=3)) to ensure positive-definitness."
        fisher = fisher + shift * I
    end

    push!(log_bg_mle, mle)
    push!(log_bg_fisher, fisher)
end

## Set up the model
model = PulsarLightcurveExtraction.spec_fourier_model(cm, sm, fg_spectral_design_matrix, bg_spectral_design_matrix, event_segment_indices, fg_exposure, bg_exposure, log_bg_mle, log_bg_fisher, fractional_variability)

## Set up the autodiff
if !use_mooncake
    @info "Using Enzyme for AD"
    adtype = AutoEnzyme(mode=Enzyme.set_runtime_activity(Enzyme.Reverse))
else
    @info "Using Mooncake for AD"
    adtype = AutoMooncake()
end

n_warmup = n_mcmc
n_total = 2 * n_mcmc

## Checkpoint file(s), one per chain. If every chain already has one on disk, we can
## resume sampling directly from the saved states and skip Pathfinder entirely.
checkpoint_paths = if checkpoint_path === nothing
    fill(nothing, n_chain)
elseif n_chain == 1
    [checkpoint_path]
else
    ["$(checkpoint_path).chain$(i)" for i in 1:n_chain]
end
resuming = checkpoint_path !== nothing && all(p -> isfile(p), checkpoint_paths)

if resuming
    @info "Found checkpoint(s) at $(checkpoint_path); resuming sampling without re-running Pathfinder."
    init_params_per_chain = fill(nothing, n_chain)
    # The metric here only bootstraps a chain that has no checkpoint; every resumed
    # chain carries its own adapted metric in its saved state, so this placeholder is
    # never actually used.
    kernel = externalsampler(
        NUTSCustomBuffer(
            target_arate;
            max_depth=max_tree_depth,
            metric=:diagonal,
            init_buffer=init_buffer,
            term_buffer=term_buffer,
            window_size=window_size,
        );
        adtype=adtype
    )
else
    ## Pathfinder initialization
    # init_scale=init_width uses Pathfinder.UniformSampler(init_width), giving U(-init_width, init_width)
    # in unconstrained space — must stay near zero to avoid LKJ coordinate singularities.
    @info "Running Pathfinder (init_scale=$(init_width)) to initialize sampler position and diagonal metric..."
    pf_result = pathfinder(model;
        init_scale=init_width,
        ndraws=max(n_chain, 4),
        adtype=adtype
    )

    # Diagonal of the LBFGS inverse-Hessian approximation in unconstrained space.
    # fit_distribution.Σ is a low-rank structured matrix (rank ≤ 2*lbfgs_memory);
    # diag() extracts the diagonal without ever allocating the O(d²) dense matrix.
    metric_diag = diag(pf_result.fit_distribution.Σ)
    @info "Pathfinder complete. Model unconstrained dimension: $(length(metric_diag))"

    ## NUTS kernel with Pathfinder-initialized diagonal mass matrix
    kernel = externalsampler(
        NUTSCustomBuffer(
            target_arate;
            max_depth=max_tree_depth,
            metric=DiagEuclideanMetric(metric_diag),
            init_buffer=init_buffer,
            term_buffer=term_buffer,
            window_size=window_size,
        );
        adtype=adtype
    )

    ## Per-chain starting positions from Pathfinder draws
    # AbstractMCMC.to_samples converts the MCMCChains.Chains in draws_transformed back to
    # ParamsWithStats objects; .params gives the constrained NamedTuple for InitFromParams.
    pf_samples = AbstractMCMC.to_samples(DynamicPPL.ParamsWithStats, pf_result.draws_transformed, model)
    init_params_per_chain = [InitFromParams(pf_samples[i].params) for i in 1:n_chain]
end

## Sample it, checkpointing periodically if requested (see run_chain, defined above).
# n_adapts must be passed explicitly: externalsampler forwards kwargs to AdvancedHMC's
# step function, which defaults to n_adapts=0 (no adaptation) if not given. Without
# this, the Pathfinder metric is never refined by WelfordVar and the step size found
# by find_good_stepsize is never updated by dual averaging.
@info "Sampling with $n_segments segments of data with $n_chain chains ($n_warmup warmup + $n_mcmc sampling steps)..."
if checkpoint_path !== nothing
    @info "Checkpointing to $(checkpoint_path) every $(checkpoint_every) iterations"
end

# Chains run as separate worker processes, so a plain ProgressMeter bar can't be
# ticked directly from inside run_chain. Instead, run_chain put!s the number of
# newly-completed iterations onto a RemoteChannel as it goes (a resume jumps by the
# checkpointed iteration count, everything after that is +1 per step); a task on the
# main process drains the channel and advances the bar. A -1 sentinel, sent from a
# `finally` so it fires even if sampling errors, tells the drain task to stop.
progress_channel = RemoteChannel(() -> Channel{Int}(64))
progress_bar = Progress(n_chain * n_total; desc="Sampling: ", showspeed=true)

# ProgressMeter's ETA/speed are elapsed_time / (counter - start) since `tinit`, i.e. a
# lifetime average — misleading here since per-iteration time varies a lot (and resumed
# chains report a big batch of already-done iterations instantly). We instead maintain our
# own exponential moving average of the per-iteration duration, with time constant `tau`
# (in units of completed iterations) of 0.1 * total iterations, and on every update fake a
# single "virtual" run at that averaged duration via `tinit` (`start` stays fixed at 0)
# so ProgressMeter's built-in ETA/speed calculation reports it directly. For irregularly
# spaced samples (each channel receipt covers `amount` iterations, not always 1 — a resume
# reports many at once), the correct continuous-time analogue of an EMA weights the new
# sample by alpha = 1 - exp(-amount / tau) instead of a fixed alpha. amount/tau is usually
# small (amount is usually 1, tau ~ 10% of the total iteration count), where 1 - exp(-x)
# loses precision to cancellation; -expm1(-x) computes it directly instead.
#
# A plain recursive EMA (ema = alpha*sample + (1-alpha)*ema_prev, seeded with ema=sample on
# the first update) implicitly treats ema_prev as a fully-weighted estimate even right at
# the start, when it's really just one (possibly atypical, e.g. JIT-inflated) sample — that
# sample's influence then lingers at full strength for a full tau-iterations before it
# decays out. We instead track the accumulated decayed weight `w_sum` alongside the
# decayed weighted sum `ema_num` and divide them (West's algorithm / Adam-style bias
# correction), so early samples are automatically discounted in proportion to how little
# accumulated weight actually backs them, and the estimate converges to new data faster.
total_iters = n_chain * n_total
tau = 0.1 * total_iters
run_start_t = time()

results = nothing
@sync begin
    @async begin
        n_done = 0
        w_sum = 0.0   # accumulated (decayed) weight
        ema_num = 0.0 # accumulated (decayed) weighted sum of sample rates
        last_t = run_start_t
        while (amount = take!(progress_channel)) >= 0
            t = time()
            sample_rate = (t - last_t) / amount
            decay = exp(-amount / tau)
            alpha = -expm1(-amount / tau) # = 1 - decay, computed without cancellation
            w_sum = decay * w_sum + alpha
            ema_num = decay * ema_num + alpha * sample_rate
            ema_rate = ema_num / w_sum
            last_t = t
            n_done += amount

            if n_done >= total_iters
                # Restore the true elapsed time so the final "Time: ..." summary reports
                # the actual wall-clock duration, not the faked-out EMA window.
                progress_bar.tinit = run_start_t
            else
                progress_bar.tinit = t - ema_rate * n_done
            end
            update!(progress_bar, n_done)
        end
    end
    @async try
        global results = if n_chain == 1
            [run_chain(model, kernel, n_warmup, n_total, checkpoint_paths[1], checkpoint_every, init_params_per_chain[1], progress_channel)]
        else
            pmap(
                (cp, ip) -> run_chain(model, kernel, n_warmup, n_total, cp, checkpoint_every, ip, progress_channel),
                checkpoint_paths,
                init_params_per_chain,
            )
        end
    finally
        put!(progress_channel, -1)
    end
end

## run_chain only keeps sampling-phase draws (warmup draws are discarded once the
## adaptation window closes), so no further splitting of warmup from sampling is needed.
sampling_chains = AbstractMCMC.chainsstack([
    AbstractMCMC.bundle_samples(draws, model, kernel, state, Turing.Inference.DEFAULT_CHAIN_TYPE)
    for (draws, state) in results
])

## Package it up
trace = from_mcmcchains(sampling_chains;
    dims=Dict(
        :dlog_total_counts => (:spec,),
        :log_total_counts => (:spec,),
        :mu_log_bg => (:spec,),
        :mu_bg => (:spec,),
        :sigma_log_bg => (:spec,),
        :cholesky_corr_log_bg => (:spec, :spec2),
        :cholesky_cov_log_bg => (:spec, :spec2),
        :cov_log_bg => (:spec, :spec2),
        :log_fg_coeff_const => (:spec,),
        :fg_coeff_const => (:spec,),
        :log_bg_raw => (:spec, :segment),
        :log_bg => (:spec, :segment),
        :bg => (:spec, :segment),
        :dfg_coeffs_cos => (:spec, :fourier),
        :dfg_coeffs_sin => (:spec, :fourier),
        :fg_coeffs_cos => (:spec, :fourier),
        :fg_coeffs_sin => (:spec, :fourier)),
    coords=Dict(
        :fourier => 1:n_fourier,
        :segment => analysis_segment_inds,
        :spec => 1:n_spec,
        :spec2 => 1:n_spec))

## Check minimum ESS (sampling draws only):
println("Minimum ESS: ", minimum(ess(trace)))

## Save the chains
to_netcdf(trace, outpath)

## Now that the final trace is safely on disk, the checkpoints are no longer needed --
## remove them so a future invocation with the same --checkpoint path starts fresh
## rather than (harmlessly, but confusingly) resuming from an already-completed run.
if checkpoint_path !== nothing
    for p in checkpoint_paths
        rm(p; force=true)
    end
end
