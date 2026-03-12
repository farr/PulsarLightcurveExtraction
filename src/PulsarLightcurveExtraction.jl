module PulsarLightcurveExtraction

using ArviZ
using Bijectors
using Colors
using DimensionalData
using Distributions
using LinearAlgebra
using Makie
using Statistics
using Turing

export logdiffexp
export PI_TO_KEV
export cos_sin_matrices, segment_indices
export event_areas
export estimate_log_bg
export estimate_bg_spec
export spectral_indices
export phase_histogram_rates
export energy_bin_areas
export spec_fourier_model
export rebin_energy
export husl_wheel
export traceplot
export median_and_bands
export foreground_background_lightcurves_segment
export flush_stderr_stdout_callback

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
    spectral_indices(event_pi, n_spec)

Given event PI values and number of spectral bins, return the spectral bin index
for each event and the bin edges in PI.
"""
function spectral_indices(event_pi, n_spec)
    min_pi = minimum(event_pi)
    max_pi = maximum(event_pi)
    bins = exp.(range(log(min_pi), stop=log(max_pi), length=n_spec+1))

    # Because we exp(range(log(...))), want to make sure the first and last bins are exactly min and max, to avoid any numerical issues with events that have PI values at the edges.
    bins[1] = min_pi
    bins[end] = max_pi 

    event_spectral_indices = searchsortedfirst.(Ref(bins[2:end]), event_pi)
    return event_spectral_indices, bins
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
    event_areas(times, pi_indices, arf_starts, arf_ends, arf_e_low, arf_e_high, arf_areas)

Given event times and PI indices, along with ARF start and end times, high
energy bounds, and area matrix, return the effective area for each event.
"""
function event_areas(times, spectral_indices, pi_bins, arf_starts, arf_ends, arf_e_low, arf_e_high, arf_areas)
    areas = Float64[]
    for i in eachindex(times)
        iarf = searchsortedfirst(arf_ends, times[i])
        @assert times[i] >= arf_starts[iarf] && times[i] <= arf_ends[iarf]
        eg_low = PI_TO_KEV*pi_bins[spectral_indices[i]]
        eg_high = PI_TO_KEV*pi_bins[spectral_indices[i]+1]
        jarf_low = searchsortedfirst(arf_e_high, eg_low) # PI to keV
        jarf_high = searchsortedfirst(arf_e_high, eg_high)
        push!(areas, mean(arf_areas[jarf_low:jarf_high, iarf]))
    end
    return areas
end

""" 
    estimate_log_bg(event_segment_indices, event_spectral_indices, segment_starts, segment_ends)

Given event segment indices, event spectral indices, and segment start and end
times, estimate the log background rate and its uncertainty in each spectral bin
and each segment using a simple counting method with a Jeffreys prior.
"""
function estimate_log_bg(event_segment_indices, event_spectral_indices, segment_starts, segment_ends)
    n_segments = length(segment_starts)
    n_spec = maximum(event_spectral_indices)
    segment_Ts = reshape(segment_ends .- segment_starts, (1, :))
    counts_per_segment = zeros(Int, n_spec, n_segments)
    for (seg, spec) in zip(event_segment_indices, event_spectral_indices)
        counts_per_segment[spec, seg] += 1
    end
    est_log_bg = log.(counts_per_segment .+ 0.5) .- log.(segment_Ts)
    est_log_bg_uncert = 1.0 ./ sqrt.(counts_per_segment .+ 0.5)
    return est_log_bg, est_log_bg_uncert
end

"""
    estimate_log_fg_const(event_segment_indices, event_spectral_indices, energy_bin_areas, segment_Ts)

Returns an estimate of the log_fg_const and associated uncertainty, attributing
all counts in each energy bin to the foreground.  This is used to scale the
sampler variables, so need not be particularly accurate.
"""
function estimate_log_fg_const(event_segment_indices, event_spectral_indices, energy_bin_areas, segment_Ts)
    n_spec = size(energy_bin_areas, 1)

    counts_per_spec = zeros(Int, n_spec)
    for (seg, spec) in zip(event_segment_indices, event_spectral_indices)
        counts_per_spec[spec] += 1
    end

    energy_bin_exposure = sum(energy_bin_areas .* reshape(segment_Ts, (1, :)), dims=2)

    est_log_fg_const = log.(counts_per_spec .+ 0.5) .- log.(energy_bin_exposure)
    est_log_fg_const_uncert = 1.0 ./ sqrt.(counts_per_spec .+ 0.5)
    return est_log_fg_const, est_log_fg_const_uncert
end

"""
    energy_bin_areas(spec_bins, segment_starts, segment_ends, arf_starts, arf_ends, arf_e_low, arf_e_high, arf_area)

Returns the effective area for each energy bin at each segement by averaging the
ARF in that segment over each energy bin.
"""
function energy_bin_areas(spec_bins, segment_starts, segment_ends, arf_starts, arf_ends, arf_e_low, arf_e_high, arf_area)
    exposures = zeros(Float64, length(spec_bins)-1, length(segment_starts))

    for i in axes(exposures, 2)
        start = segment_starts[i]
        stop = segment_ends[i]

        iarf_start = searchsortedfirst(arf_ends, start)
        iarf_stop = searchsortedfirst(arf_ends, stop)
        @assert iarf_start == iarf_stop "Segment $i overlaps multiple ARF intervals, which is not currently supported."

        for j in axes(exposures, 1)
            pi_low = spec_bins[j]
            pi_high = spec_bins[j+1]

            e_low = PI_TO_KEV * pi_low
            e_high = PI_TO_KEV * pi_high

            jarf_low = searchsortedfirst(arf_e_high, e_low)
            jarf_high = searchsortedfirst(arf_e_high, e_high)

            exposures[j, i] += mean(arf_area[jarf_low:jarf_high, iarf_start])
        end
    end

    exposures
end

raw"""
    spec_fourier_model(design_matrix, event_segment_indices, event_spectral_indices, event_sensitive_areas, segment_Ts, energy_bin_exposure, est_log_bg, est_log_bg_uncert, fg_scale)

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
@model function spec_fourier_model(cos_design_matrix, sin_design_matrix, event_segment_indices, event_spectral_indices, segment_Ts, energy_bin_areas; fractional_variability=0.1)
    @assert size(cos_design_matrix, 2) == size(sin_design_matrix, 2) "Cosine and sine design matrices should have the same number of columns."

    n_counts, n_fourier = size(cos_design_matrix)
    n_seg = length(segment_Ts)
    n_eg_bin = maximum(event_spectral_indices)

    T = sum(segment_Ts)
    cts_per_second = n_counts / T
    
    exposure = sum(energy_bin_areas .* reshape(segment_Ts, (1, :)))
    cts_per_second_per_cm2 = n_counts / exposure

    mu_mu_log_bg ~ Normal(log(cts_per_second), 4)
    sigma_mu_log_bg ~ Exponential(2)

    mu_log_fg_const ~ Normal(log(cts_per_second_per_cm2), 4)
    sigma_log_fg_const ~ Exponential(2)

    mu_log_bg_uncentered = Vector{Float64}(undef, n_eg_bin)
    mu_log_bg = Vector{Float64}(undef, n_eg_bin)
    for i in eachindex(mu_log_bg)
        mu_log_bg_uncentered[i] ~ Normal(0, 1)
        mu_log_bg[i] := mu_mu_log_bg + sigma_mu_log_bg * mu_log_bg_uncentered[i]
    end
    
    sigma_log_bg = Vector{Float64}(undef, n_eg_bin)
    for i in eachindex(sigma_log_bg)
        sigma_log_bg[i] ~ Exponential(1)
    end

    corr_chol ~ LKJCholesky(n_eg_bin, 2.0)
    L_cov := Diagonal(sigma_log_bg) * corr_chol.L
    cov_log_bg := L_cov * L_cov'

    log_fg_coeff_const_uncentered = Vector{Float64}(undef, n_eg_bin)
    log_fg_coeff_const = Vector{Float64}(undef, n_eg_bin)
    fg_coeff_const = Vector{Float64}(undef, n_eg_bin)
    for i in eachindex(fg_coeff_const)
        log_fg_coeff_const_uncentered[i] ~ Normal(0, 1)
        log_fg_coeff_const[i] := mu_log_fg_const + sigma_log_fg_const * log_fg_coeff_const_uncentered[i]
        fg_coeff_const[i] := exp(log_fg_coeff_const[i])
    end
    
    log_bg_uncentered = Matrix{Float64}(undef, n_eg_bin, n_seg)
    log_bg = Matrix{Float64}(undef, n_eg_bin, n_seg)
    bg = Matrix{Float64}(undef, n_eg_bin, n_seg)
    for j in axes(log_bg_uncentered, 2)
        for i in axes(log_bg_uncentered, 1)
            log_bg_uncentered[i, j] ~ Normal(0, 1)
        end
        lbg_j = mu_log_bg .+ (L_cov * log_bg_uncentered[:, j])
        for i in axes(log_bg, 1)
            log_bg[i, j] := lbg_j[i]
            bg[i, j] := exp(log_bg[i, j])
        end
    end

    dsigma_fg ~ Exponential(1)
    sigma_fg := fractional_variability * cts_per_second_per_cm2 * dsigma_fg

    dfg_coeffs_cos = Matrix{Float64}(undef, n_eg_bin, n_fourier)
    dfg_coeffs_sin = Matrix{Float64}(undef, n_eg_bin, n_fourier)
    fg_coeffs_cos = Matrix{Float64}(undef, n_eg_bin, n_fourier)
    fg_coeffs_sin = Matrix{Float64}(undef, n_eg_bin, n_fourier)
    for i in 1:n_eg_bin
        for j in 1:n_fourier
            dfg_coeffs_cos[i, j] ~ Normal(0, 1)
            dfg_coeffs_sin[i, j] ~ Normal(0, 1)

            fg_coeffs_cos[i,j] := sigma_fg * dfg_coeffs_cos[i,j]
            fg_coeffs_sin[i,j] := sigma_fg * dfg_coeffs_sin[i,j]
        end
    end

    for i in eachindex(event_spectral_indices)
        fg_rate = fg_coeff_const[event_spectral_indices[i]]
        for k in 1:n_fourier
            fg_rate += cos_design_matrix[i, k] * fg_coeffs_cos[event_spectral_indices[i], k] + sin_design_matrix[i, k] * fg_coeffs_sin[event_spectral_indices[i], k]
        end
        fg_rate *= energy_bin_areas[event_spectral_indices[i], event_segment_indices[i]]

        bg_rate += bg[event_spectral_indices[i], event_segment_indices[i]]

        rate = fg_rate + bg_rate
        if rate <= 0 # Mathematical error if rate <= 0 because of log(...).  Really should ensure *foreground* rate is positive, but that can cause issues with sampling.
            # Cannot have negative rate!
            Turing.@addlogprob! -Inf
        else
            Turing.@addlogprob! log(rate)
        end
    end

    for i in eachindex(segment_Ts)
        for j in axes(bg, 1)
            Turing.@addlogprob! -bg[j,i] * segment_Ts[i] # BG counts from segment i and energy bin j.
        end
    end

    for j in eachindex(segment_Ts)
        for i in axes(energy_bin_areas, 1)
            Turing.@addlogprob! -fg_coeff_const[i] * segment_Ts[j] * energy_bin_areas[i,j]
        end
    end
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
    foreground_background_lightcurves_segment(trace, segment, phases, spec_bins_pi, segment_start, segment_stop, arf_start, arf_stop, arf_e_low, arf_e_high, arf_response)

Given a trace of posterior samples, segment index, array of phases (spanning
``[0,1]``), energy bin edges in PI, segment start and stop times, ARF start and
stop times, ARF energy bin edges, and ARF response matrix, compute the
foreground, background, and total lightcurves for the specified segment by
marginalizing over the posterior samples.  The foreground lightcurve is computed
by summing the constant part of the foreground and the variable part of the
foreground (the Fourier components) across energy bins, weighted by the exposure
for each energy bin.  The background lightcurve is computed as a constant across
phase (since the model assumes a constant background rate in each segment) by
summing over energy bins weighted by the background spectrum.  The total
lightcurve is the sum of the foreground and background lightcurves.

The lightcurves will be in units of counts per second as a function of phase.

Returns a tuple of `(fg_lc, bg_lc, total_lc)`
"""
function foreground_background_lightcurves_segment(trace, segment, phases, energy_bin_areas)
    cm, sm = cos_sin_matrices(phases, size(trace.posterior.fg_coeffs_cos, :fourier))
    cm = DimArray(cm, (:phases => phases, :fourier => 1:size(cm,2)))
    sm = DimArray(sm, (:phases => phases, :fourier => 1:size(sm,2)))

    energy_bin_areas = DimArray(vec(energy_bin_areas), dims(trace.posterior.fg_coeff_const, :energy))

    variable_fg_lc = dropdims(sum(@d((cm .* trace.posterior.fg_coeffs_cos .+ sm .* trace.posterior.fg_coeffs_sin) .* energy_bin_areas), dims=(:fourier, :energy)), dims=(:fourier, :energy))
    const_fg_lc = dropdims(sum(@d(trace.posterior.fg_coeff_const .* energy_bin_areas), dims=:energy), dims=:energy)
    fg_lc = @d const_fg_lc .+ variable_fg_lc

    # No exposures in the background
    bg_lc = dropdims(sum(@d(trace.posterior.bg[segment=At(segment)] .* DimArray(ones(length(phases)), :phases => phases)), dims=:energy), dims=:energy)

    total_lc = @d fg_lc .+ bg_lc

    return fg_lc, bg_lc, total_lc
end

function flush_stderr_stdout_callback(args...; kwargs...)
    flush(stdout)
    flush(stderr)
end

end # module PulsarLightcurveExtraction
