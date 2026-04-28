module PulsarLightcurveExtraction

using ArviZ
using BSplineKit
using ProgressLogging
using Colors
using DimensionalData
using Distributions
using LinearAlgebra
using Makie
using Statistics
using Turing

export logdiffexp
export PI_TO_KEV
export cos_sin_matrices
export construct_bspline_basis
export construct_bsplane_spectral_bases
export segment_indices
export spectral_design_matrices
export foreground_background_exposure
export phase_histogram_rates
export spec_fourier_model
export rate_threshold_from_segments
export background_mle_per_segment
export husl_wheel
export traceplot
export median_and_bands
export foreground_background_lightcurves_segment

raw"""
    softplus(x[, eps=1.0])

Returns the softplus of `x/eps`, defined as `eps*log(1 + exp(x/eps))`, but
computed in a numerically-stable way.  The `eps` parameter controls the
"softness" of the function; as `eps` approaches zero, the softplus approaches
the ReLU function `max(0, x)`.
"""
function softplus(x)
    softplus(x, one(x))
end

function softplus(x, eps)
    eps * log1p(exp(-abs(x/eps))) + max(x, zero(x))
end

raw"""
    logdiffexp(x, y)

Returns ``\log\left( \exp(x) - \exp(y) \right)`` but computed in a
numerically-stable way.
"""
function logdiffexp(x, y)
    x + log1p(-exp(y-x))
end

""" 
    PI_TO_KEV

Conversion factor from PI channel to keV.
"""
const PI_TO_KEV = 0.01

""" 
    cos_sin_matrices(phases, n_fourier)

Given a vector of phases and number of Fourier harmonics, return design matrices
of cosine and sine terms for use in Fourier series modeling.
"""
function cos_sin_matrices(phases, n_fourier)
    cos_matrix = hcat([cos.(2*pi.*phases.*k) for k in 1:n_fourier]...)
    sin_matrix = hcat([sin.(2*pi.*phases.*k) for k in 1:n_fourier]...)
    return (cos_matrix, sin_matrix)
end

"""
    construct_bspline_basis(e_min, e_max, n_spec, spec_order)

Returns the B-spline basis object with the given energy limits, number of basis
elements, and order.  The knots are placed logarithmically in energy.
"""
function construct_bspline_basis(e_min, e_max, n_spec, spec_order)
    knots = exp.(range(log(e_min), log(e_max), length=n_spec-2))
    return BSplineBasis(BSplineOrder(spec_order), knots)
end

"""
    construct_spline_spectral_bases(e_min, e_max, n_spec, spec_order, e_low, e_high, arf_response, rmf_response)

Constructs spline bases for spectral modeling given energy bounds, number of
spectral bins, spline order, and instrument responses (ARF and RMF).  Returns
`(spline_to_pi_foreground, spline_to_pi_background)`, both of shape `(num_pi,
num_basis, num_segments)`, which give the mapping from spline basis coefficients
to PI indices within each ARF/RMF time segment.  The first matrix returned
incorporates the ARF effective areas, so is appropriate for converting a
spline-based spectral model with coefficients expressed in counts per second per
square cm per keV to expected counts per second in each PI bin; the second
matrix returned incorporates only the RMF, so is appropriate for converting a
spline-based spectral model with coefficients expressed in counts per second per
keV to expected counts per second in each PI bin.  The latter is used to model
the background spectrum, which need not respond to the effective area because it
need not follow the optical path of the foreground photons.
"""
function construct_spline_spectral_bases(e_min, e_max, n_spec, spec_order, e_low, e_high, arf_response, rmf_response)
    egs = 0.5 .* (e_low .+ e_high)
    
    basis = construct_bspline_basis(e_min, e_max, n_spec, spec_order)

    spline_to_energy = zeros(Float32, length(egs), length(basis))
    for (i, e) in enumerate(egs)
        j, vals = basis(e)
        if j >= spec_order
            spline_to_energy[i, j:-1:j-spec_order+1] .= vals
        end # Otherwise outside the range of the basis functions, so all zero.
    end
    spline_to_energy = spline_to_energy .* reshape(e_high .- e_low, (:, 1))

    combined_response = rmf_response .* reshape(arf_response, (1, size(arf_response)...))

    spline_to_pi_combined = stack([combined_response[:,:,t] * spline_to_energy for t in axes(combined_response, 3)], dims=3)
    spline_to_pi = stack([rmf_response[:,:,t] * spline_to_energy for t in axes(rmf_response, 3)], dims=3)

    return (spline_to_pi_combined, spline_to_pi)
end

""" 
    segment_indices(times, segment_starts, segment_ends)

Given a vector of event times and segment start and end times, return the index
of the segment each event belongs to.
"""
function segment_indices(times, segment_starts, segment_ends)
    segment_indices = searchsortedfirst.((segment_starts,), times) .- 1
    @assert all((times .< segment_ends[segment_indices]) .&& (times .>= segment_starts[segment_indices])) "Some events do not fall within any segment."

    return segment_indices
end

"""
    spectral_design_matrices(times, event_pi, fg_spline_basis, bg_spline_basis, spec_starts, spec_ends)

Construct design matrices for foreground and background spectral models given
event times, PI channels, spline bases, and segment start and end times. Returns
`(fg_design_matrix, bg_design_matrix)` where each has shape `(n_events,
n_spline_basis)`, and the `i`th row of each corresponds to the spline basis
evaluated at the PI channel of the `i`th event, with the appropriate segment
index determined by the event time.  The `fg` matrix incorporates the ARF
effective area for the source, while the `bg` matrix does not.
"""
function spectral_design_matrices(times, event_pi, fg_spline_basis, bg_spline_basis, spec_starts, spec_ends)
    n_events = length(times)
    n_spec = size(fg_spline_basis, 2)

    fg_design_matrix = zeros(Float64, n_events, n_spec)
    bg_design_matrix = zeros(Float64, n_events, n_spec)

    for i in 1:n_events
        segment_bin = searchsortedfirst(spec_starts, times[i]) - 1
        @assert times[i] >= spec_starts[segment_bin] && times[i] <= spec_ends[segment_bin] "Event time does not fall within any segment bin."

        fg_design_matrix[i, :] .= fg_spline_basis[event_pi[i], :, segment_bin]
        bg_design_matrix[i, :] .= bg_spline_basis[event_pi[i], :, segment_bin]
    end

    return (fg_design_matrix, bg_design_matrix)
end

"""
    foreground_background_exposure(pi_min, pi_max, segment_starts, segment_ends, arf_starts, arf_ends, fg_spline_basis, bg_spline_basis)

Returns `(fg_exposure, bg_exposure)` where `fg_exposure` is a vector of size
`n_spec` that gives the total exposure to each of the spectral basis elements.
For the foreground elements, this is the total exposure in square-cm seconds;
for the background is it the exposure per segment in seconds.  This is used to
compute the expected number of counts from the constant term in the foreground,
and the expected number of counts from the background in each segment.
"""
function foreground_background_exposure(pi_min, pi_max, segment_starts, segment_ends, arf_starts, arf_ends, fg_spline_basis, bg_spline_basis)
    n_segments = length(segment_starts)
    n_spec = size(fg_spline_basis, 2)

    fg_exposure = zeros(Float64, n_spec)
    bg_exposure = zeros(Float64, n_spec, n_segments)

    summed_fg_basis = dropdims(sum(fg_spline_basis[pi_min:pi_max, :, :], dims=1), dims=1)
    summed_bg_basis = dropdims(sum(bg_spline_basis[pi_min:pi_max, :, :], dims=1), dims=1)

    for i in 1:n_segments
        segment_start = segment_starts[i]
        segment_end = segment_ends[i]

        T = segment_end - segment_start

        arf_bin = searchsortedfirst(arf_ends, segment_start)
        @assert segment_start >= arf_starts[arf_bin] && segment_end <= arf_ends[arf_bin] "Segment does not fall within any ARF bin."

        fg_exposure .= fg_exposure .+ T .* summed_fg_basis[:, arf_bin]
        bg_exposure[:, i] = T .* summed_bg_basis[:, arf_bin]
    end

    return (fg_exposure, bg_exposure)
end

"""
    background_maxl_and_fisher(bg_spectral_design_matrix, bg_exposure)

Returns the maximum-likelihood estimate and fisher information matrix for the
log of the background spectral parameters given the background spectral design
matrix and the background exposure, assuming a pure-background model.
"""
function background_maxl_and_fisher(bg_spectral_design_matrix, bg_exposure)
    function dlogl_dlogb(log_bg)
        rates = bg_spectral_design_matrix * exp.(log_bg)
        [(sum(bg_spectral_design_matrix[:, j] ./ rates) - bg_exposure[j])*exp(log_bg[j]) for j in axes(log_bg, 1)]
    end

    function dlogl_dlogb2(log_bg)
        rates = bg_spectral_design_matrix * exp.(log_bg)

        [-sum((bg_spectral_design_matrix[:,j] .* bg_spectral_design_matrix[:,k]) ./ (rates .* rates)) * exp(log_bg[j]) * exp(log_bg[k]) for j in axes(log_bg, 1), k in axes(log_bg, 1)]
    end

    log_bg_old = -Inf .* ones(size(bg_exposure, 1))
    log_bg = log(size(bg_spectral_design_matrix, 1) / sum(bg_exposure)) .* ones(size(bg_exposure, 1))
    iterations = 0
    while sum(abs.(log_bg_old .- log_bg)) > 1e-3 && iterations < 100
        log_bg_old = log_bg
        resid = dlogl_dlogb(log_bg)
        fisher = -dlogl_dlogb2(log_bg) + Diagonal(1e-6 * ones(size(log_bg)))
        log_bg = log_bg .+ fisher \ resid
        iterations += 1
    end

    (log_bg, -dlogl_dlogb2(log_bg))
end

"""
    background_mle_per_segment(bg_spectral_design_matrix, event_segment_indices, bg_exposure)

For each observing segment, compute the MLE of the log background spectral
coefficients under a pure-background Poisson model, and the marginal
uncertainties from the Fisher information.  Returns `(log_bg_hat, s_hat)`,
both of shape `(n_spec, n_seg)`.  Empty segments have `s_hat[:, j]` set to
`Inf`, which causes the partially non-centered parameterization in
`spec_fourier_model` to reduce to the standard non-centered form for those
segments.
"""
function background_mle_per_segment(bg_spectral_design_matrix, event_segment_indices, bg_exposure)
    n_spec, n_seg = size(bg_exposure)
    log_bg_hat = zeros(Float64, n_spec, n_seg)
    s_hat = fill(Inf, n_spec, n_seg)

    @progress "Background MLE per segment" for j in 1:n_seg
        seg_sel = event_segment_indices .== j
        try
            log_bg_j, fisher_j = background_maxl_and_fisher(
                bg_spectral_design_matrix[seg_sel, :],
                bg_exposure[:, j]
            )
            log_bg_hat[:, j] = log_bg_j
            s_hat[:, j] = sqrt.(diag(inv(fisher_j)))                
        catch e
                @info "Caught exception $(e), covariance will be set to Inf for segment $(j)"
        end
    end

    return log_bg_hat, s_hat
end

"""
    phase_histogram_rates(phases, exposure_time)

Given photon `phases` in `[0, 1]` and total `exposure_time` (seconds), build a
histogram using a Scott-like rule specialized to unit-width support, replacing
`σ` with half the central 68% span (`0.68 / 2`):

`h_bin = 3.5 * 0.68 / 2 / n^(1/3)`, `n_bin = ceil(Int, 1 / h_bin)`.

Returns `(bin_edges, rates, rate_uncertainty)` where `bin_edges` has length
`n_bin + 1` and `rates` has length `n_bin`, in counts per second per unit phase
for each bin (i.e. rate conditioned on being in that phase interval).
"""
function phase_histogram_rates(phases, exposure_time)
    @assert exposure_time > 0 "Exposure time must be positive."
    @assert !isempty(phases) "At least one phase value is required."
    @assert all(0 .<= phases .<= 1) "All phases must lie in [0, 1]."

    n = length(phases)
    h_bin = 3.5 * 0.68 / 2 / n^(1 / 3)
    n_bin = ceil(Int, 1 / h_bin)

    bin_edges = collect(range(0.0, 1.0; length=n_bin + 1))
    counts = zeros(Int, n_bin)

    for p in phases
        # Include p == 1 in the final bin for right-edge closure.
        i = p == 1 ? n_bin : searchsortedfirst(bin_edges, p) - 1
        counts[i] += 1
    end

    bin_widths = diff(bin_edges)
    rates = counts ./ (exposure_time .* bin_widths)
    rate_uncertainty = sqrt.(counts) ./ (exposure_time .* bin_widths)
    return bin_edges, rates, rate_uncertainty
end

"""
    rate_threshold_from_segments(segment_start, segment_stop, pi_min, pi_max, event_segment_indices)

Compute a softplus threshold rate (counts per second per PI bin) suitable for
use in `spec_fourier_model`.  For each segment, the background count rate is
estimated as `(n_events + 0.5) / T` (the +0.5 ensures positivity even for
empty segments).  The threshold is 10% of the minimum per-segment rate
distributed uniformly across active PI bins, ensuring it is negligible relative
to any physically meaningful signal.
"""
function rate_threshold_from_segments(segment_start, segment_stop, pi_min, pi_max, event_segment_indices)
    n_segments = length(segment_start)
    n_pi_active = pi_max - pi_min + 1

    counts_per_segment = zeros(Int, n_segments)
    for seg in event_segment_indices
        counts_per_segment[seg] += 1
    end

    T_segments = segment_stop .- segment_start
    rates_per_segment = (counts_per_segment .+ 0.5) ./ T_segments

    return 0.1 * minimum(rates_per_segment) / n_pi_active
end

raw"""
    spec_fourier_model(design_matrix, event_segment_indices, event_spectral_indices, event_sensitive_areas, segment_Ts, energy_bin_exposure, est_log_bg, est_log_bg_uncert, fg_scale)

WARNING: this docstring is out of date and needs updating!!
A spectral-photometric model for a pulsar phasecurve with a varying background.

The background is modeled as a constant rate of detected photons in each
observing segment (observations are segmented according to discrete observing
intervals on the ISS).  The foreground is modeled as a linear combination of the
basis functions that comprise the columns of the design matrix.  Both background
and foreground are also decomposed spectrally, by probability-per-energy-bin
(the background is assumed to be spectrally constant---not a good assumption,
but it likely doesn't matter much for the fit, and makes things a lot
easier---while each foreground Fourier component gets its own spectral
decomposition). So, in segment ``i``, the background detection rate in energy
bin ``j`` is given by 

``\frac{\mathrm{d} N}{\mathrm{d} t} = B_i p^{\mathrm{bg}}_j``

for all photons that arrive in segment ``i``.  The foreground detection rate
varies by photon arrival phase relative to the radio pulse of the neutron star,
and is given at the arrival time of photon ``i`` by 

``\frac{\mathrm{d} N}{\mathrm{d} t_i} = \mathrm{Area}_i \left( A_\mathrm{const}
p^{\mathrm{fg},\mathrm{const}}_j + \sum_{k} M_{ik} A_k p^{\mathrm{fg},k}_j
\right) ``

for basis function coefficients ``A_k`` plus a constant term,
``A_\mathrm{const}``.  ``\mathrm{Area}_i`` is the effective area for the arrival
of the ``i``th photon.  

The model assumes that the basis functions integrate to zero over the pulsar
phase (as would be expected for a Fourier decomposition, for example).  The
`energy_bin_exposure` is the time-and-mean-area associated with each energy bin
to compute the expected number of counts from the constant term in the
foreground; so that the total expected photon count over all segments is given
by 

``N_\mathrm{exp} = \sum_i B_i T_i + A_\mathrm{const} \sum_j
p^{\mathrm{fg},\mathrm{const}}_j E_j``

with ``T_i`` the observing time of segment ``i``, and ``E_j`` the exposure for
energy bin ``j``.  The model fits the background count rates ``B_i``, the
foreground basis function coefficients ``A_k``, and the foreground and
background spectra ``p^{\mathrm{bg}/\mathrm{fg}}_j`` as parameters.  The fit is
performed using a hierarchical model with partial pooling (sometimes also called
a "random effects" model in a regression context), where the background rates
``B_i`` are given a normal prior with a mean and s.d. that are, in turn,
parameters; and the foreground basis function coefficients ``A_k`` are given a
zero-mean normal prior with a s.d. that is, in turn, a model parameter (zero
mean because we imagine that the design matrix columns are the sin and cos
components of a Fourier basis, and we want an isotropic prior in phase).
Allowing the mean and s.d. of these priors to be parameters lets the model learn
the typical background and the dispersion of the background and foreground
coefficients and use this information to inform the estimates of the set of
coefficients collectively.

The phase-varying parts of the model are encoded in the `design_matrix` which
should have shape `n_photons, n_components`, where `n_components` is the number
of basis functions used to model the phase curve (e.g., Fourier components; see
`cos_sin_matrices` above).  If the design matrix incorporates the sensitive area
of the detector for each photon (e.g. by multiplying the corresponding row by
the effective area at the PI energy for that photon), then the foreground
coefficients will be in units of counts per square cm per second, and the
`fg_scale` parameter should be set according to the typical amount of foreground
that is reasonable in those units.

`event_segment_indices` gives the background segment in which each photon falls.

`event_spectral_indices` gives the energy bin in which each photon falls.

`segment_Ts` gives the total observing time for each segment.

`est_log_bg` and `est_log_bg_uncert` are estimates of the log-rate of the
background and the associated uncertainty in each segment.  Since for many
segments the background count rate is very high, the ``B_i`` parameters are very
precisely measured; these estimates are used to help guide the sampler to the
narrow peak of the posterior density and help it to more efficiently explore in
this tightly-constrained region.  See `estimate_log_bg` above to produce these
quantities.

`est_bg_spec` and `est_bg_spec_uncert` function similarly for the background
spectrum, which is also extremely well-determined (by many thousands of counts
per bin); see `estimate_bg_spec` above to produce these quantities.

`mu_log_fg_const` and `sigma_log_fg_const` are used to set a prior on the
constant part of the foreground, and should be estimates of the expected log
counts per square cm per second in the constant part of the foreground, and the
uncertainty on that estimate.  This is used to help guide the sampler to the
region of parameter space where the foreground is reasonably consistent with the
data, which can help it to more efficiently explore the posterior density.

`fg_scale` is used to set a prior on the foreground dispersion parameter,
`sigma_fg` (if, for example, the design matrix carries units of effective area,
with sin and cos modulations according to a Fourier basis, then the ``A_k`` have
units of counts per time per area, and `fg_scale` should be set according to the
typical amount of foreground that is reasonable).

The model that is returned is suitable for sampling with Turing.jl samplers.
"""
@model function spec_fourier_model(cos_design_matrix, sin_design_matrix, fg_spectral_design_matrix, bg_spectral_design_matrix, event_segment_indices, fg_exposure, bg_exposure, fractional_variability, rate_threshold)
    n_counts, n_fourier = size(cos_design_matrix)
    n_spec, n_seg = size(bg_exposure)

    @assert size(sin_design_matrix) == (n_counts, n_fourier) "Sine design matrix must have size n_counts, n_fourier."
    @assert size(fg_spectral_design_matrix) == (n_counts, n_spec) "Foreground spectral design matrix must have size n_counts, n_spec."
    @assert size(bg_spectral_design_matrix) == (n_counts, n_spec) "Background spectral design matrix must have size n_counts, n_spec."
    @assert size(event_segment_indices) == (n_counts,) "Event segment indices must have size n_counts."
    @assert size(fg_exposure) == (n_spec,) "Foreground exposure must have size n_spec."

    total_bg_exposure = sum(bg_exposure)
    total_fg_exposure = sum(fg_exposure)

    est_bg_rate = n_counts / total_bg_exposure
    est_fg_rate = n_counts / total_fg_exposure

    mu_log_bg = Vector{Float64}(undef, n_spec)
    mu_bg = Vector{Float64}(undef, n_spec)
    @inbounds for i in 1:n_spec
        mu_log_bg[i] ~ Normal(log(est_bg_rate), 2.0)
        mu_bg[i] := exp(mu_log_bg[i])
    end

    sigma_log_bg = Vector{Float64}(undef, n_spec)
    @inbounds for i in 1:n_spec
        sigma_log_bg[i] ~ truncated(Normal(0.0, 1.0), 0.0, Inf)
    end

    log_fg_coeff_const = Vector{Float64}(undef, n_spec)
    fg_coeff_const = Vector{Float64}(undef, n_spec)
    @inbounds for i in 1:n_spec
        log_fg_coeff_const[i] ~ Normal(log(est_fg_rate), 2.0)
        fg_coeff_const[i] := exp(log_fg_coeff_const[i])
    end

    log_bg = Matrix{Float64}(undef, n_spec, n_seg)
    bg = Matrix{Float64}(undef, n_spec, n_seg)
    @inbounds for j in 1:n_seg
        @inbounds for i in 1:n_spec
            log_bg[i, j] ~ Normal(mu_log_bg[i], sigma_log_bg[i])
            bg[i,j] := exp(log_bg[i,j])
        end
    end

    dsigma_fg ~ truncated(Normal(0.0, 1.0), 0.0, Inf)
    sigma_fg := fractional_variability * est_fg_rate * dsigma_fg

    dfg_coeffs_cos = Matrix{Float64}(undef, n_spec, n_fourier)
    dfg_coeffs_sin = Matrix{Float64}(undef, n_spec, n_fourier)
    fg_coeffs_cos = Matrix{Float64}(undef, n_spec, n_fourier)
    fg_coeffs_sin = Matrix{Float64}(undef, n_spec, n_fourier)
    @inbounds for j in 1:n_fourier
        @inbounds for i in 1:n_spec
            dfg_coeffs_cos[i, j] ~ Normal(0.0, 1.0)
            dfg_coeffs_sin[i, j] ~ Normal(0.0, 1.0)
            fg_coeffs_cos[i, j] := sigma_fg * dfg_coeffs_cos[i, j]
            fg_coeffs_sin[i, j] := sigma_fg * dfg_coeffs_sin[i, j]
        end
    end

    fg_cos_mat = fg_spectral_design_matrix * fg_coeffs_cos  # n_counts × n_fourier
    fg_sin_mat = fg_spectral_design_matrix * fg_coeffs_sin  # n_counts × n_fourier
    fg_rates = fg_spectral_design_matrix * fg_coeff_const .+
               dropdims(sum(fg_cos_mat .* cos_design_matrix .+
                            fg_sin_mat .* sin_design_matrix, dims=2), dims=2)

    bg_rates = dropdims(sum(bg_spectral_design_matrix' .* bg[:, event_segment_indices], dims=1), dims=1)

    Turing.@addlogprob! sum(log.(softplus.(fg_rates .+ bg_rates, rate_threshold)))

    ex_cts = dot(fg_coeff_const, fg_exposure) + dot(bg, bg_exposure)
    Turing.@addlogprob! -ex_cts

    return
end

"""
    husl_wheel(n)

Produce a color wheel of `n` "distinguishable" colors in the LCHuv color space,
which is designed to be perceptually uniform.  This can be used for plotting
multiple chains in a traceplot, for example.
"""
function husl_wheel(n)
    return [LCHuv(65, 90, h) for h in range(0, stop=360, length=n+1)][1:end-1]
end

"""
    traceplot(chain; var_names=nothing)

Produce a traceplot of the posterior samples in `chain`, which is an ArviZ trace
object.  If `var_names` is provided, only plot those variables; otherwise, plot
all variables in the posterior.  The traceplot shows the marginal density of
each variable across all chains (left column) and the trace of each variable
across iterations for each chain (right column), with different colors for
different chains.
"""
function traceplot(chain; var_names=nothing)
    p = chain.posterior

    if var_names !== nothing
        vars = var_names
    else
        vars = keys(p)
    end

    f = Figure(size=(800, 200*length(vars)))
    for (i, var) in enumerate(vars)
        zs = p[var]
        c = dims(zs, :chain)
        ds = otherdims(zs, (:chain, :draw))
        
        all_ds = (c, ds...)

        zs_merge = mergedims(zs, all_ds => :merged_chain)
        n = size(zs_merge, :merged_chain)

        adens = Axis(f[i,1], xlabel=String(var), palette=(color = husl_wheel(n), patchcolor = husl_wheel(n),))
        atrace = Axis(f[i,2], xlabel="Iteration", ylabel=String(var), palette=(color = husl_wheel(n), patchcolor = husl_wheel(n),))

        for c in dims(zs_merge, :merged_chain)
            z = vec(zs_merge[merged_chain=At(c)])

            # h_est = 3.5 * (quantile(z, 0.84) - quantile(z, 0.16)) / 2 / length(z)^(1/3) # Scott's rule, replacing std with equivalent quantile range
            # n_bins = ceil(Int, (maximum(z) - minimum(z)) / h_est)

            # hist!(adens, vec(z); bins=n_bins, label=nothing, norm=:pdf)
            density!(adens, vec(z); label=nothing)
            lines!(atrace, vec(z); label=nothing)
        end
    end

    return f
end

"""
    median_and_bands(array; q=((0.16, 0.84), (0.025, 0.975)))

Given an array of posterior samples with dimensions `:chain` and `:draw`,
compute the median and credible intervals across those dimensions.  The `q`
argument specifies the quantiles to compute for the credible intervals; by
default, it computes the 68% credible interval (between the 16th and 84th
percentiles) and the 95% credible interval (between the 2.5th and 97.5th
percentiles).  The output is a tuple of `(median, lower_uppers)` where `median`
is the median across chains and draws, and `lower_uppers` is a tuple of tuples
of `(lower, upper)` for each pair of quantiles specified in `q`.
"""
function median_and_bands(array; q=((0.16, 0.84), (0.025, 0.975)))
    m = dropdims(median(array, dims=(:chain, :draw)), dims=(:chain, :draw))
    lower_uppers = map(q) do qq
        lower = dropdims(mapslices(x -> quantile(vec(x), qq[1]), array; dims=(:chain, :draw)), dims=(:chain, :draw))
        upper = dropdims(mapslices(x -> quantile(vec(x), qq[2]), array; dims=(:chain, :draw)), dims=(:chain, :draw))
        return (lower, upper)
    end
    return (m, lower_uppers)
end

"""
    foreground_background_lightcurves_segment(trace, phases, segment_index, pi_min, pi_max, segment_starts, segment_ends, arf_starts, arf_ends, fg_spline_basis, bg_spline_basis)

Returns `(fg_lightcurve, bg_lightcurve, total_lightcurve)` for the segment with
index `segment_index` given the posterior samples in `trace`, the photon
`phases`, the segment start and end times, the ARF start and end times, and the
foreground and background spline bases.  The lightcurves are returned as
DimArrays.  The `phases` argument should be a DimArray with a dimension of
`:phase`; the returned lightcurves will have the same dimension.

`fg_lightcurve` gives the foreground contribution to the lightcurve,
`bg_lightcurve` gives the background contribution, and `total_lightcurve` gives
the total (foreground + background) contribution.  The lightcurves are in units
of counts per second as a function of phase.
"""
function foreground_background_lightcurves_segment(trace, phases, segment_index, pi_min, pi_max, segment_starts, segment_ends, arf_starts, arf_ends, fg_spline_basis, bg_spline_basis)
    T = segment_ends[segment_index] - segment_starts[segment_index]
    p = trace.posterior
    
    fg_exposure_array, bg_exposure_array = foreground_background_exposure(pi_min, pi_max, [segment_starts[segment_index]], [segment_ends[segment_index]], arf_starts, arf_ends, fg_spline_basis, bg_spline_basis)
    bg_exposure_single = bg_exposure_array[:, 1]

    # We actually don't want the *total* exposure---we want the exposure *per
    # time* so that the lightcurve is in units of counts per second.
    fg_exposure_array ./= T
    bg_exposure_single ./= T

    fg_exposure = DimArray(fg_exposure_array, dims(p, :spec))
    bg_exposure = DimArray(bg_exposure_single, dims(p, :spec))

    cmm, smm = cos_sin_matrices(phases, size(p, :fourier))
    cm = DimArray(cmm, (Dim{:phase}(phases), dims(p, :fourier)))
    sm = DimArray(smm, (Dim{:phase}(phases), dims(p, :fourier)))

    fg_lightcurve_variable = dropdims(sum(@d(p.fg_coeffs_cos .* cm .+ p.fg_coeffs_sin .* sm), dims=:fourier), dims=:fourier)
    fg_lightcurve = dropdims(sum(@d(fg_exposure .* (p.fg_coeff_const .+ fg_lightcurve_variable)), dims=:spec), dims=:spec)

    bg_lightcurve_unsummed = @d bg_exposure .* p.bg[segment=At(segment_index)] .+ 0.0 .* cm[fourier=At(1)] # Add zero times a phase array to it has a phase dimension
    bg_lightcurve = dropdims(sum(bg_lightcurve_unsummed, dims=:spec), dims=:spec)

    total_lightcurve = @d fg_lightcurve .+ bg_lightcurve

    (fg_lightcurve, bg_lightcurve, total_lightcurve)
end

end # module PulsarLightcurveExtraction
