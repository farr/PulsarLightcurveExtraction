module PulsarLightcurveExtraction

using ArviZ
using Bijectors
using Distributions
using LinearAlgebra
using Turing

export PI_TO_KEV
export cos_sin_matrices, segment_indices
export event_areas
export estimate_log_bg
export estimate_bg_spec
export spectral_indices
export spec_fourier_model

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
    @assert all(times .< segment_ends[segment_indices])

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
    bins = range(min_pi, stop=max_pi, length=n_spec+1)
    event_spectral_indices = searchsortedfirst.(Ref(bins[2:end]), event_pi)
    return event_spectral_indices, bins
end

raw"""
    spec_fourier_model(design_matrix, event_segment_indices, event_spectral_indices, segment_Ts, est_log_bg, est_log_bg_uncert, est_bg_spec, est_bg_spec_uncert, fg_scale)

A spectral-photometric model for a pulsar phasecurve with a varying background.

The background is modeled as a constant rate of detected photons in each
observing segment (observations are segmented according to discrete observing
intervals on the ISS).  The foreground is modeled as a linear combination of the
basis functions that comprise the columns of the design matrix.  Both background
and foreground are also decomposed spectrally, by probability-per-energy-bin.
So, in segment ``i``, the background detection rate in energy bin ``j`` is given
by 

``\frac{\mathrm{d} N}{\mathrm{d} t} = B_i p^{\mathrm{bg}}_j``

for all photons that arrive in segment ``i``.  The foreground detection rate
varies by photon arrival phase relative to the radio pulse of the neutron star,
and is given at the arrival time of photon ``i`` by 

``\frac{\mathrm{d} N}{\mathrm{d} t_i} = p^{\mathrm{fg}}_j \sum_{k} M_{ik} A_k``

for basis function coefficients ``A_k``.  The model assumes that the basis
functions integrate to zero over the pulsar phase (as would be expected for a
Fourier decomposition, for example), so that the total expected photon count
over all segments is given by 

``N_\mathrm{exp} = \sum_i B_i T_i``

with ``T_i`` the observing time of segment ``i``.  The model fits the background
count rates ``B_i``, the foreground basis function coefficients ``A_k``, and the
foreground and background spectra ``p^{\mathrm{bg}/\mathrm{fg}}_j`` as
parameters.  The fit is performed using a hierarchical model with partial
pooling (sometimes also called a "random effects" model in a regression
context), where the background rates ``B_i`` are given a normal prior with a
mean and s.d. that are, in turn, parameters; and the foreground basis function
coefficients ``A_k`` are given a zero-mean normal prior with a s.d. that is, in
turn, a model parameter (zero mean because we imagine that the design matrix
columns are the sin and cos components of a Fourier basis, and we want an
isotropic prior in phase).  Allowing the mean and s.d. of these priors to be
parameters lets the model learn the typical background and the dispersion of the
background and foreground coefficients and use this information to inform the
estimates of the set of coefficients collectively.

The phase-varying parts of the model are encoded in the `design_matrix` which
should have shape `n_photons, n_components`, where `n_components` is the number
of basis functions used to model the phase curve (e.g., Fourier components; see
`cos_sin_matrices` above).  

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

`fg_scale` is used to set a prior on the foreground dispersion parameter,
`sigma_fg` (if, for example, the design matrix carries units of effective area,
with sin and cos modulations according to a Fourier basis, then the ``A_k`` have
units of counts per time per area, and `fg_scale` should be set according to the
typical amount of foreground that is reasonable).

The model that is returned is suitable for sampling with Turing.jl samplers.
"""
@model function spec_fourier_model(design_matrix, event_segment_indices, event_spectral_indices, segment_Ts, est_log_bg, est_log_bg_uncert, est_bg_spec, est_bg_spec_uncert, fg_scale)
    mlbg = mean(est_log_bg)
    slbg = std(est_log_bg)
    n_spec = maximum(event_spectral_indices)

    # Transform variables so sampler sees unit-scale, but parameters have
    # physical scale.  No Jacobians needed because the transformation is based
    # only on data.  Was trying to do this with Bijectors.Shift and
    # Bijectors.scale, but Mooncake errored out on the AutoDiff with that for
    # some reason.  TODO: bring this back to Bijectors.shift and Bijectors.scale
    # if possible.
    dmu_log_bg ~ Normal(0,1)
    mu_log_bg := mlbg + dmu_log_bg * (5*slbg / sqrt(length(segment_Ts)))
    scaled_sigma_log_bg ~ Exponential(1)
    sigma_log_bg := 2 * scaled_sigma_log_bg * slbg # Wider scale for sigma by a bit.

    sigma_fg ~ transformed(Exponential(1), Bijectors.Scale(fg_scale))

    fg_coeffs ~ filldist(transformed(Normal(0, 1), Bijectors.Scale(sigma_fg)), size(design_matrix, 2)) # Zero mean because we want an isotropic prior to give uniform phase coverage.

    # log_bg_segment = est_log_bg .+ est_log_bg_uncert .* log_dbg_segment (i.e. log_dbg_segment measures the number of sigma away from the estimate)
    # If log_bg_segment ~ Normal(mu_log_bg, sigma_log_bg), then log_dbg_segment ~ Normal((mu_log_bg .- est_log_bg) ./ est_log_bg_uncert, sigma_log_bg ./ est_log_bg_uncert)
    log_dbg_segment ~ arraydist([Normal((mu_log_bg - est_log_bg[i]) / est_log_bg_uncert[i], sigma_log_bg / est_log_bg_uncert[i]) for i in 1:length(segment_Ts)])
    log_bg_segment := est_log_bg .+ est_log_bg_uncert .* log_dbg_segment
    bg_segment := exp.(log_bg_segment)

    # Spectrum: probability of event being in each bin given bg or fg; flat, single-count Dirichlet priors.
    fg_spec ~ Dirichlet(fill(1.0, maximum(event_spectral_indices)))

    # This is a trick: we get the bijector to/from the probability space and the
    # unconstrained space.  Then in the unconstrained space, we put a
    # N(est_bg_spec, 5*est_bg_spec_uncert), and transform back to probability
    # space.  That way the sampler sees variables that are ~unit scale even
    # though the background spectrum is very precisely measured.
    d = Dirichlet(fill(1.0, maximum(event_spectral_indices)))
    b = bijector(d)
    bi = inverse(b)
    dbg_spec ~ arraydist([Normal(0, 1) for _ in 1:(n_spec-1)]) # Last element is determined by the simplex constraint.
    bg_spec := bi(est_bg_spec .+ 5 .* est_bg_spec_uncert .* dbg_spec) 

    rate_at_events = bg_segment[event_segment_indices] .* bg_spec[event_spectral_indices] .+ (design_matrix * fg_coeffs) .* fg_spec[event_spectral_indices]
    if any(rate_at_events .<= 0)
        Turing.@addlogprob! -Inf
    else
        Turing.@addlogprob! sum(log.(rate_at_events)) - sum(bg_segment .* segment_Ts)
    end
end

end # module PulsarLightcurveExtraction
