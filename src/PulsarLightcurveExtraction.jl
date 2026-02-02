module PulsarLightcurveExtraction

using ArviZ
using CategoricalArrays
using DimensionalData
using Distributions
using DynamicPPL
using LinearAlgebra
using Turing

export PI_TO_KEV
export cos_sin_matrices, segment_indices
export event_areas

const PI_TO_KEV = 0.01

function cos_sin_matrices(phases, n_fourier)
    cos_matrix = hcat([cos.(2*pi.*phases.*k) for k in 1:n_fourier]...)
    sin_matrix = hcat([sin.(2*pi.*phases.*k) for k in 1:n_fourier]...)
    return (cos_matrix, sin_matrix)
end

function segment_indices(times, segment_starts, segment_ends)
    segment_indices = searchsortedfirst.((segment_starts,), times) .- 1
    @assert all(times .< segment_ends[segment_indices])

    return segment_indices
end

function event_areas(times, pi_indices, arf_starts, arf_ends, arf_e_high, arf_areas)
    areas = Float64[]
    for i in eachindex(times)
        iarf = searchsortedfirst(arf_ends, times[i])
        @assert times[i] >= arf_starts[iarf] && times[i] <= arf_ends[iarf]
        jarf = searchsortedfirst(arf_e_high, PI_TO_KEV*pi_indices[i]) # PI to keV
        push!(areas, arf_areas[jarf, iarf])
    end
    return areas
end

function estimated_background_rates_and_counts(segment_indices, segment_starts, segment_ends)
    nsegments = length(segment_starts)
    counts = zeros(Int, nsegments)

    for i in segment_indices
        counts[i] += 1
    end

    return (counts ./ (segment_ends .- segment_starts), counts)
end

function binned_counts_fourier_model_mean_cholesky_precision(bin_centers, bin_counts, bin_counts_uncertainty, n_fourier)
    cos_matrix, sin_matrix = cos_sin_matrices(bin_centers, n_fourier)
    const_matrix = ones(length(bin_centers))

    M = hcat(const_matrix, cos_matrix, sin_matrix)

    sigma_squared = bin_counts_uncertainty.^2

    CInvM = (M ./ reshape(sigma_squared, (length(sigma_squared), 1)))
    AInv = Hermitian(M' * CInvM)
    AInvCholesky = cholesky(AInv)
    a = AInvCholesky \ (M' * (bin_counts ./ sigma_squared))

    (a, AInvCholesky, M)
end

@model function varying_background_fourier_model(segment_indices, segment_starts, segment_ends, cos_matrix, sin_matrix, lc_cos_matrix, lc_sin_matrix; period_amp_prior_frac=0.15)
    ncounts, nfourier = size(cos_matrix)

    segment_Ts = segment_ends .- segment_starts
    bg_est = basic_bg_estimate(segment_indices, segment_Ts)
    zero_segments = bg_est .== 0.0
    bc_est = zeros(nfourier)
    bs_est = zeros(nfourier)
    
    for _ in 1:10
        bg_est, bc_est, bs_est = nr_step(segment_indices, segment_Ts, cos_matrix, sin_matrix, bg_est, bc_est, bs_est)
    end
    I_bg = fisher_bg_diag(segment_indices, segment_Ts, cos_matrix, sin_matrix, bg_est, bc_est, bs_est)
    I_bc, I_bs = fisher_betas_diag(segment_indices, segment_Ts, cos_matrix, sin_matrix, bg_est, bc_est, bs_est)

    sigma_bg = 1.0 ./ sqrt.(I_bg)
    sigma_bc = 1.0 ./ sqrt.(I_bc)
    sigma_bs = 1.0 ./ sqrt.(I_bs)

    # Clean up the zero-count segments
    bg_est[zero_segments] .= 0.5 ./ segment_Ts[zero_segments]
    sigma_bg[zero_segments] .= sqrt(0.5) ./ segment_Ts[zero_segments]

    sigma_beta_scale = std(vcat(bc_est, bs_est))

    log_bg_est = log.(bg_est)
    log_bg_est_scale = sigma_bg ./ bg_est

    mu_log_bg_est = mean(log_bg_est)
    sigma_log_bg_est = std(log_bg_est)
    
    nseg = length(segment_starts)
    @assert length(segment_ends) == nseg

    T = sum(segment_ends .- segment_starts)
    mean_rate = ncounts/T

    mu_log_bg_scaled ~ Flat()
    mu_log_bg = mu_log_bg_est + mu_log_bg_scaled * sigma_log_bg_est / sqrt(nseg)
    Turing.@addlogprob! logpdf(Normal(mu_log_bg_est, 10*sigma_log_bg_est), mu_log_bg)

    sigma_log_bg_scaled ~ FlatPos(0.0)
    sigma_log_bg = sigma_log_bg_est * sigma_log_bg_scaled
    Turing.@addlogprob! logpdf(Exponential(1), sigma_log_bg)

    log_bg_segment_scaled ~ filldist(Flat(), nseg)
    log_bg_segment = log_bg_est .+ log_bg_est_scale .* log_bg_segment_scaled
    Turing.@addlogprob! sum(logpdf.((Normal(mu_log_bg, sigma_log_bg),), log_bg_segment))

    bg_segment = exp.(log_bg_segment)
    total_bg = sum(bg_segment .* (segment_ends .- segment_starts))

    sigma_beta_scaled ~ FlatPos(0.0)
    sigma_beta = sigma_beta_scale * sigma_beta_scaled
    Turing.@addlogprob! logpdf(Exponential(1), sigma_beta)

    beta_cos_scaled ~ filldist(Flat(), nfourier)
    beta_sin_scaled ~ filldist(Flat(), nfourier)
    beta_cos = bc_est .+ beta_cos_scaled .* sigma_bc
    beta_sin = bs_est .+ beta_sin_scaled .* sigma_bs
    Turing.@addlogprob! sum(logpdf.((Normal(0, sigma_beta),), vcat(beta_cos, beta_sin)))

    rate_at_events = bg_segment[segment_indices] .+ cos_matrix*beta_cos .+ sin_matrix*beta_sin

    if any(rate_at_events .< 0)
        Turing.@addlogprob! -Inf
    else
        Turing.@addlogprob! sum(log.(rate_at_events)) - total_bg
    end

    lc = lc_cos_matrix*beta_cos .+ lc_sin_matrix*beta_sin

    (mu_log_bg=mu_log_bg, sigma_log_bg=sigma_log_bg, bg_segment=bg_segment, sigma_beta=sigma_beta, beta_cos=beta_cos, beta_sin=beta_sin, lc=lc)
end

@model function varying_background_spectral_fourier_model(pi_indices, segment_indices, segment_starts, segment_ends, cos_matrix, sin_matrix, lc_cos_matrix, lc_sin_matrix; period_amp_prior_frac=0.15)
    nseg = length(segment_starts)
    @assert length(segment_ends) == nseg

    ncounts, nfourier = size(cos_matrix)
    @assert size(sin_matrix) == (ncounts, nfourier)

    nchannels = maximum(pi_indices)

    T = sum(segment_ends .- segment_starts)
    mean_rate = ncounts/T

    # Half a count for regularization
    bg_est_counts = 0.5 .* ones(nseg)
    for si in segment_indices
        bg_est_counts[si] += 1
    end
    bg_est_rates = bg_est_counts ./ (segment_ends .- segment_starts)
    bg_est_rates_uncertainty = sqrt.(bg_est_counts) ./ (segment_ends .- segment_starts)

    log_bg_est_rates = log.(bg_est_rates)
    log_bg_est_rates_uncertainty = bg_est_rates_uncertainty ./ bg_est_rates

    mu_log_bg ~ Normal(log(mean_rate), 1)
    sigma_log_bg ~ Exponential(1)

    log_bg_segment_scaled ~ filldist(Flat(), nseg)
    log_bg_segment = log_bg_est_rates .+ log_bg_segment_scaled .* log_bg_est_rates_uncertainty
    # Prior is N(mu_log_bg, sigma_log_bg) on log_bg_segment.  Because the
    # transformation from log_bg_segment to the sampled variable
    # log_bg_segment_scaled is constant (it depends only on the data), we don't
    # need to account for it in the sampling
    Turing.@addlogprob! sum(logpdf.((Normal(mu_log_bg, sigma_log_bg),), log_bg_segment))

    bg_segment = exp.(log_bg_segment)
    total_bg = sum(bg_segment .* (segment_ends .- segment_starts))

    sigma_beta_scaled ~ Exponential(1)
    sigma_beta = sigma_beta_scaled .* mean_rate .* period_amp_prior_frac
    
    beta_cos ~ filldist(Normal(0, sigma_beta), nfourier)
    beta_sin ~ filldist(Normal(0, sigma_beta), nfourier)

    bg_spec ~ Dirichlet(ones(nchannels))
    fg_spec ~ filldist(Dirichlet(ones(nchannels)), nfourier)

    fg_spec_matrix = fg_spec[pi_indices, :]

    rate_at_events = bg_segment[segment_indices].*bg_spec[pi_indices] .+ (cos_matrix .* fg_spec_matrix)*beta_cos .+ (sin_matrix .* fg_spec_matrix)*beta_sin

    if any(rate_at_events .< 0)
        Turing.@addlogprob! -Inf
    else
        Turing.@addlogprob! sum(log.(rate_at_events)) - total_bg
    end

    lc = lc_cos_matrix*beta_cos .+ lc_sin_matrix*beta_sin

    (bg_segment=bg_segment, lc=lc, sigma_beta=sigma_beta) # beta_cos=beta_cos, beta_sin=beta_sin, 
end

# Here are some utility functions that can help with sampling
"""
    parameters_from_arviz(model, idata; unconstrained=true)

Given a model and some InferenceData, return a matrix of the vectorized
parameter samples from the model.  The matrix will have shape `(ndim, ndraws,
nchain)`.

If `unconstrained` is `true` then the parameters will be in the unconstrained
parameter space; otherwise they will be in the constrained parameter space.
"""
function parameters_from_arviz(model, idata; unconstrained=true)
    posterior = idata.posterior

    vi = DynamicPPL.VarInfo(model)

    model_vars = keys(vi)

    # ArviZ posterior variable names (normalize to Symbols)
    function get_vectorized_values(c, d)
        for name in model_vars
            vi[name] = posterior[Symbol(name)][chain=At(c), draw=At(d)]
        end
        if unconstrained
            vi_linked = DynamicPPL.link(vi, model)
        else
            vi_linked = vi
        end
        DynamicPPL.getindex_internal(vi_linked, :)
    end

    stack([get_vectorized_values(c, d) for d in dims(posterior, :draw), c in dims(posterior, :chain)])
end

"""
    external_sampler_from_arviz(model, idata; adtype=Turing.AutoMooncake(; config=nothing), target_accept=0.8)

Return an external sampler instance setup for use with Turing's `sample`
function, that uses a mass matrix / metric estimated from the given model and
samples, and the given AD backend.
"""
function external_sampler_from_arviz(model, idata; adtype=Turing.AutoMooncake(; config=nothing), target_accept=0.8)
    uc_params = parameters_from_arviz(model, idata; unconstrained=true)

    # The metric is the *inverse* of the mass matrix, which should be the
    # covariance.
    diag_metric = AdvancedHMC.DiagEuclideanMetric(dropdims(var(uc_params; dims=(2,3)), dims=(2,3)))

    nuts = AdvancedHMC.NUTS(target_accept; metric=diag_metric)
    Turing.externalsampler(nuts; adtype=adtype)
end

"""
    initial_values_from_arviz(model, idata, chains=1)

Return a vector of initial values or a vector of vectors of initial values (if
`chains>1`) for the given model and InferenceData, sampled randomly from the
draws and chains in the InferenceData.
"""
function initial_values_from_arviz(model, idata)
    c_params = parameters_from_arviz(model, idata; unconstrained=false)

    (_, ndraw, nchain) = size(c_params)
    c_params[:, rand(1:ndraw), rand(1:nchain)]
end

function initial_values_from_arviz(model, idata, chains)
    c_params = parameters_from_arviz(model, idata; unconstrained=false)

    (_, ndraw, nchain) = size(c_params)
    [c_params[:, rand(1:ndraw), rand(1:nchain)] for _ in 1:chains]
end

# Some code for semi-analyitic optimization of the likelihood function
function rates_at_arrival(segment_indices, segment_Ts, cos_matrix, sin_matrix, bg_segment, beta_cos, beta_sin)
    bg_part = bg_segment[segment_indices]
    fg_part = cos_matrix*beta_cos .+ sin_matrix*beta_sin
    return bg_part .+ fg_part
end

function basic_bg_estimate(segment_indices, segment_Ts)
    nseg = length(segment_Ts)
    counts = zeros(Int, nseg)

    for i in segment_indices
        counts[i] += 1
    end

    rates = counts ./ segment_Ts

    return rates
end

function dlogl_dbg(segment_indices, segment_Ts, cos_matrix, sin_matrix, bg_segment, beta_cos, beta_sin)
    rates = rates_at_arrival(segment_indices, segment_Ts, cos_matrix, sin_matrix, bg_segment, beta_cos, beta_sin)

    per_segment_bg_term = zeros(length(segment_Ts))
    for (si, r) in zip(segment_indices, rates)
        per_segment_bg_term[si] += 1/r
    end

    .- segment_Ts .+ per_segment_bg_term
end

function dlogl_dbetas(segment_indices, segment_Ts, cos_matrix, sin_matrix, bg_segment, beta_cos, beta_sin)
    rates = rates_at_arrival(segment_indices, segment_Ts, cos_matrix, sin_matrix, bg_segment, beta_cos, beta_sin)

    per_fourier_cos_term = zeros(length(beta_cos))
    per_fourier_sin_term = zeros(length(beta_sin))
    for (i, r) in enumerate(rates)
        per_fourier_cos_term .+= (cos_matrix[i, :] ./ r)
        per_fourier_sin_term .+= (sin_matrix[i, :] ./ r)
    end

    per_fourier_cos_term, per_fourier_sin_term
end

function fisher_bg_diag(segment_indices, segment_Ts, cos_matrix, sin_matrix, bg_segment, beta_cos, beta_sin)
    rates = rates_at_arrival(segment_indices, segment_Ts, cos_matrix, sin_matrix, bg_segment, beta_cos, beta_sin)

    I = zeros(length(segment_Ts))

    for (si, r) in zip(segment_indices, rates)
        I[si] += 1/r^2
    end

    I
end

function fisher_betas_diag(segment_indices, segment_Ts, cos_matrix, sin_matrix, bg_segment, beta_cos, beta_sin)
    rates = rates_at_arrival(segment_indices, segment_Ts, cos_matrix, sin_matrix, bg_segment, beta_cos, beta_sin)

    nfourier = length(beta_cos)
    I_cos = zeros(nfourier)
    I_sin = zeros(nfourier)

    for (i, r) in enumerate(rates)
        I_cos .+= (cos_matrix[i, :] ./ r).^2
        I_sin .+= (sin_matrix[i, :] ./ r).^2
    end

    I_cos, I_sin
end

function nr_step(segment_indices, segment_Ts, cos_matrix, sin_matrix, bg_segment, beta_cos, beta_sin)
    d_bg = dlogl_dbg(segment_indices, segment_Ts, cos_matrix, sin_matrix, bg_segment, beta_cos, beta_sin)
    d_bc, d_bs = dlogl_dbetas(segment_indices, segment_Ts, cos_matrix, sin_matrix, bg_segment, beta_cos, beta_sin)
    
    I_bg = fisher_bg_diag(segment_indices, segment_Ts, cos_matrix, sin_matrix, bg_segment, beta_cos, beta_sin)
    I_bc, I_bs = fisher_betas_diag(segment_indices, segment_Ts, cos_matrix, sin_matrix, bg_segment, beta_cos, beta_sin)

    dx_bg = d_bg ./ I_bg
    dx_bc = d_bc ./ I_bc
    dx_bs = d_bs ./ I_bs

    bg_segment .+ dx_bg, beta_cos .+ dx_bc, beta_sin .+ dx_bs
end

end # module PulsarLightcurveExtraction
