module PulsarLightcurveExtraction

using ArviZ
using Bijectors
using Colors
using DimensionalData
using Distributions
using LinearAlgebra
using Makie
using Turing

export PI_TO_KEV
export cos_sin_matrices, segment_indices
export event_areas
export estimate_log_bg
export estimate_bg_spec
export spectral_indices
export energy_bin_exposure
export spec_fourier_model
export rebin_energy
export husl_wheel
export traceplot

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
    event_areas(times, pi_indices, arf_starts, arf_ends, arf_e_low, arf_e_high, arf_areas)

Given event times and PI indices, along with ARF start and end times, high
energy bounds, and area matrix, return the effective area for each event.
"""
function event_areas(times, pi_indices, arf_starts, arf_ends, arf_e_low, arf_e_high, arf_areas)
    areas = Float64[]
    for i in eachindex(times)
        iarf = searchsortedfirst(arf_ends, times[i])
        @assert times[i] >= arf_starts[iarf] && times[i] <= arf_ends[iarf]
        eg = PI_TO_KEV*pi_indices[i]
        jarf = searchsortedfirst(arf_e_high, eg) # PI to keV
        @assert eg >= arf_e_low[jarf] && eg <= arf_e_high[jarf]
        push!(areas, arf_areas[jarf, iarf])
    end
    return areas
end

""" 
    estimate_log_bg(event_times, segment_starts, segment_ends)

Given event times and segment start and end times, estimate the log background
rate and its uncertainty in each segment using a simple counting method with
a Jeffreys prior.
"""
function estimate_log_bg(event_times, segment_starts, segment_ends)
    n_segments = length(segment_starts)
    segment_Ts = segment_ends .- segment_starts
    counts_per_segment = zeros(Int, n_segments)
    for t in event_times
        iseg = searchsortedfirst(segment_ends, t)
        counts_per_segment[iseg] += 1
    end
    est_log_bg = log.(counts_per_segment .+ 0.5) .- log.(segment_Ts)
    est_log_bg_uncert = 1.0 ./ sqrt.(counts_per_segment .+ 0.5)
    return est_log_bg, est_log_bg_uncert
end

"""
    estimate_bg_spec(event_spectral_indices)

From the energy bin index of each event, estimate the mean and uncertainty of
the posterior over background spectral parameters.

Returns `(mu, sigma)` where `mu` is an estimated mean vector and `sigma` the
estimated s.d. in the unconstrained parameter space of the Dirichlet
distribution for the background spectrum.
"""
function estimate_bg_spec(event_spectral_indices)
    esi = event_spectral_indices

    # Put half a count in bins to start, for regularization (not that we will need it).
    bg_spec_counts = zeros(maximum(event_spectral_indices)) .+ 0.5
    for eind in esi
        bg_spec_counts[eind] += 1
    end

    # Get an empirical estimate of the mean and s.d. of the background spectrum in
    # the unconstrained parameter space of the Dirichlet distribution.
    bg_dist = Dirichlet(bg_spec_counts)
    b = bijector(bg_dist)
    samples = [b(rand(bg_dist)) for _ in 1:10000]

    est_bg_spec = mean(samples)
    est_bg_spec_uncert = std(samples)

    return est_bg_spec, est_bg_spec_uncert
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
    energy_bin_exposure(spec_bins, segment_starts, segment_ends, arf_starts, arf_ends, arf_e_low, arf_e_high, arf_area)

Returns the exposure (time * mean effective area) for each energy bin, computed
by integrating over the observing segments and the ARF.  This is used to compute
the expected number of counts from the constant term in the foreground, which is
given by the sum over energy bins of the foreground constant amplitude times the
foreground constant spectrum times the energy bin exposure.
"""
function energy_bin_exposure(spec_bins, segment_starts, segment_ends, arf_starts, arf_ends, arf_e_low, arf_e_high, arf_area)
    exposures = zeros(Float64, length(spec_bins)-1)

    for i in eachindex(segment_starts)
        start = segment_starts[i]
        stop = segment_ends[i]

        iarf_start = searchsortedfirst(arf_ends, start)
        iarf_stop = searchsortedfirst(arf_ends, stop)
        @assert iarf_start == iarf_stop "Segment $i overlaps multiple ARF intervals, which is not currently supported."

        for j in eachindex(exposures)
            pi_low = spec_bins[j]
            pi_high = spec_bins[j+1]

            e_low = PI_TO_KEV * pi_low
            e_high = PI_TO_KEV * pi_high

            jarf_low = searchsortedfirst(arf_e_high, e_low)
            jarf_high = searchsortedfirst(arf_e_high, e_high)

            exposures[j] += (stop - start) * mean(arf_area[jarf_low:jarf_high, iarf_start])
        end
    end

    exposures
end

raw"""
    spec_fourier_model(design_matrix, event_segment_indices, event_spectral_indices, event_sensitive_areas, segment_Ts, energy_bin_exposure, est_log_bg, est_log_bg_uncert, est_bg_spec, est_bg_spec_uncert, mu_log_fg_const, sigma_log_fg_const, fg_scale)

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
@model function spec_fourier_model(design_matrix, event_segment_indices, event_spectral_indices, event_sensitive_areas, segment_Ts, energy_bin_exposure, est_log_bg, est_log_bg_uncert, est_bg_spec, est_bg_spec_uncert, fg_scale)
    @assert size(design_matrix, 2) % 2 == 0 "Design matrix should have an even number of columns, with the first half being cosine terms and the second half sine terms."

    lbg = mean(est_log_bg)
    lbg_scale = 1 / sqrt(length(segment_Ts)) # Uncertainty on the mean log background rate estimate, which we use to set the scale of the prior on the mean log background rate.

    log_slbg = log(std(est_log_bg))
    log_slbg_scale = sqrt(2) / sqrt(length(segment_Ts)) # Uncertainty on the standard deviation of the log background rate estimate, which we use to set the scale of the prior on the standard deviation of the log background rate.

    lfg = log(size(design_matrix,1) / mean(energy_bin_exposure)) # Assume a flat spectrum per energy bin, and attribute all counts to the fg (this is just for scaling)
    lfg_scale = 1/sqrt(size(design_matrix,1)) # Uncertainty on the log of the foreground rate estimate, assuming all counts are from the foreground and ignoring the background (this is just for scaling)

    n_spec = maximum(event_spectral_indices)
    n_fourier = round(Int, size(design_matrix, 2) / 2)

    # We want to put a broad, N(mlbg, 2) prior on the mean log background rate
    # (i.e. 1-sigma uncertainty is a factor of 10).  But the sampler wants to
    # see unit-scale variables, centered around zero.  So we shift and scale,
    # defining mu_log_bg = mlbg + dmu_log_bg * lbg_scale.  If mu_log_bg ~
    # N(mlbg, 2), then dmu_log_bg ~ N(0, 2/lbg_scale).
    dmu_log_bg ~ Normal(0, 2/lbg_scale)
    mu_log_bg := lbg + dmu_log_bg * lbg_scale # So mu_log_bg ~ N(mlbg, 2), which is what we want.

    # We want to put a broad Exp(2) prior on sigma_log_bg; but the sampler wants
    # to see unit-scale variables, centered around zero.  So we define
    # sigma_log_bg = exp(log_sigma_log_bg), and then log_sigma_log_bg = log_slbg
    # + dlog_sigma_log_bg * log_slbg_scale.  If sigma_log_bg ~ Exp(1), then
    #   dlog_sigma_log_bg is Exp(1) * sigma_log_bg * log_slbg_scale.  
    dlog_sigma_log_bg ~ Flat()
    log_sigma_log_bg = log_slbg + dlog_sigma_log_bg * log_slbg_scale
    sigma_log_bg := exp(log_sigma_log_bg) 
    Turing.@addlogprob! logpdf(Exponential(1), sigma_log_bg) + log_sigma_log_bg + log(log_slbg_scale)

    dsigma_fg ~ Exponential(1)
    sigma_fg := fg_scale * dsigma_fg

    # Here, again, we need some shifting-and-scaling.  We want a broad N(lfg, 2)
    # prior on the constant part of the foreground (1-sigma ~ factor-of-ten),
    # but we want the sampler to see a unit-scale variable centered around zero.
    # So: log_fg_coeff_const = lfg + dlog_fg_coeff_const * lfg_scale.  If log_fg_coeff_const ~ N(lfg, 2), then dlog_fg_coeff_const ~ N(0, 2/lfg_scale).
    dlog_fg_coeff_const ~ Normal(0, 2/lfg_scale)
    log_fg_coeff_const := lfg + dlog_fg_coeff_const * lfg_scale
    fg_coeff_const := exp(log_fg_coeff_const)

    dfg_coeffs = Vector{Float64}(undef, size(design_matrix, 2))
    for k in eachindex(dfg_coeffs)
        dfg_coeffs[k] ~ Normal(0, 1)
    end
    fg_coeffs := sigma_fg .* dfg_coeffs # Exactly equivalent implied prior for fg_coeffs.
    
    # Same implied prior as the original arraydist parameterization; sampled in a
    # scalar loop for Enzyme compatibility.
    log_dbg_segment = Vector{Float64}(undef, length(segment_Ts))
    for i in eachindex(log_dbg_segment)
        m_i = (mu_log_bg - est_log_bg[i]) / est_log_bg_uncert[i]
        s_i = sigma_log_bg / est_log_bg_uncert[i]
        log_dbg_segment[i] ~ Normal(m_i, s_i)
    end
    log_bg_segment := est_log_bg .+ est_log_bg_uncert .* log_dbg_segment
    bg_segment := exp.(log_bg_segment)

    # This is a trick: we get the bijector to/from the probability space and the
    # unconstrained space.  In the unconstrained space, we shift-and-scale so
    # that the sampler should see a unit-scale distribution near zero (the bg
    # spectrum and the average fg spectrum are tightly constrained, so this
    # re-scaling is necessary).  But we want the prior ultimately to be
    # Dirichlet(1-vector); the prior on the shifted-and-scaled variables is
    # Dirirchlet(1-vector)(bg_spec) * dirichlet_jacobian(bg_spec) *
    # est_bg_spec_uncert
    d = Dirichlet(fill(1.0, maximum(event_spectral_indices)))
    b = bijector(d)
    bi = inverse(b)
    dbg_spec = Vector{Float64}(undef, n_spec-1) # Last element is determined by the simplex constraint.
    for i in eachindex(dbg_spec)
        dbg_spec[i] ~ Flat() # Will put in a prior manually.
    end
    bg_spec := bi(est_bg_spec .+ est_bg_spec_uncert .* dbg_spec)
    Turing.@addlogprob! logpdf(d, bg_spec) - logabsdetjac(b, bg_spec) + sum(log.(est_bg_spec_uncert))

    # For the variable part of the fg spectrum, we just put a broad, flat prior.
    fg_spec ~ filldist(Dirichlet(fill(1.0, n_spec)), n_fourier)

    # For the constant part of the fg spectrum, we use the same trick as the for the background spectrum.
    dfg_spec_const = Vector{Float64}(undef, n_spec-1)
    for i in eachindex(dfg_spec_const)
        dfg_spec_const[i] ~ Flat()
    end
    fg_spec_const := bi(est_bg_spec .+ est_bg_spec_uncert .* dfg_spec_const)
    Turing.@addlogprob! logpdf(d, fg_spec_const) - logabsdetjac(b, fg_spec_const) + sum(log.(est_bg_spec_uncert))

    has_nonpositive_rate = false
    log_events = zero(eltype(bg_segment))
    for i in eachindex(event_segment_indices)
        segi = event_segment_indices[i]
        speci = event_spectral_indices[i]
        areai = event_sensitive_areas[i]

        rate_i = fg_coeff_const * fg_spec_const[speci]
        for k in 1:n_fourier
            spec_w = fg_spec[speci, k]
            rate_i += design_matrix[i, k] * fg_coeffs[k] * spec_w
            rate_i += design_matrix[i, n_fourier + k] * fg_coeffs[n_fourier + k] * spec_w
        end
        rate_i *= areai

        rate_i += bg_segment[segi] * bg_spec[speci]
        
        if rate_i <= 0
            has_nonpositive_rate = true
        else
            log_events += log(rate_i)
        end
    end

    if has_nonpositive_rate
        # Cannot have negative rate!
        Turing.@addlogprob! -Inf
    else
        bg_count = sum(bg_segment .* segment_Ts)
        fg_count = fg_coeff_const * sum(fg_spec_const .* energy_bin_exposure)

        Turing.@addlogprob! log_events - bg_count - fg_count
    end
end

"""
    rebin_energy(fspec; bins_per=2, keep_partial=true)

Merge `bins_per` neighboring energy bins by summing probabilities.
Output has fewer `energy` bins and still sums to 1 along energy (if all bins are kept).

- `bins_per=2` merges pairs of bins.
- `keep_partial=true` keeps a final smaller bin if length(energy) is not divisible by `bins_per`.
"""
function rebin_energy(fspec; bins_per::Int=2, keep_partial::Bool=true)
    bins_per > 0 || error("`bins_per` must be positive.")

    A = parent(fspec)
    ds = dims(fspec)

    edim = findfirst(d -> Symbol(name(d)) == :energy, ds)
    edim === nothing && error("No `energy` dimension found.")

    nE = size(A, edim)
    nBout = keep_partial ? cld(nE, bins_per) : fld(nE, bins_per)

    outsz = collect(size(A))
    outsz[edim] = nBout
    B = zeros(eltype(A), Tuple(outsz))

    # Get old energy coordinates (bin centers)
    Eold = collect(lookup(ds[edim]))
    Enew = similar(Eold, nBout)

    for b in 1:nBout
        lo = (b - 1) * bins_per + 1
        hi = min(b * bins_per, nE)

        # Sum grouped bins into one output bin
        selectdim(B, edim, b) .= sum(selectdim(A, edim, lo:hi); dims=edim)

        # New bin center = mean of merged centers
        Enew[b] = mean(@view Eold[lo:hi])
    end

    newdims = ntuple(i -> i == edim ? Dim{:energy}(Enew) : ds[i], length(ds))
    return DimArray(B, newdims)
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

    if var_names != nothing
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

end # module PulsarLightcurveExtraction
