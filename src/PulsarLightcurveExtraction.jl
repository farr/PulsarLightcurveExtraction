module PulsarLightcurveExtraction

using CategoricalArrays
using Distributions
using LinearAlgebra
using Turing

export cos_sin_matrices, segment_indices
export estimated_background_rates_and_counts
export varying_background_fourier_model, binned_counts_fourier_model_mean_cholesky_precision
export varying_background_spectral_fourier_model

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
    nseg = length(segment_starts)
    @assert length(segment_ends) == nseg

    ncounts, nfourier = size(cos_matrix)

    T = sum(segment_ends .- segment_starts)
    mean_rate = ncounts/T

    mu_log_bg ~ Normal(log(mean_rate), 1)
    sigma_log_bg ~ Exponential(1)

    log_bg_segment ~ filldist(Normal(mu_log_bg, sigma_log_bg), nseg) 
    bg_segment = exp.(log_bg_segment)
    total_bg = sum(bg_segment .* (segment_ends .- segment_starts))

    sigma_beta_scaled ~ Exponential(1)
    sigma_beta = sigma_beta_scaled .* mean_rate .* period_amp_prior_frac
    
    beta_cos ~ filldist(Normal(0, sigma_beta), nfourier)
    beta_sin ~ filldist(Normal(0, sigma_beta), nfourier)

    rate_at_events = bg_segment[segment_indices] .+ cos_matrix*beta_cos .+ sin_matrix*beta_sin

    if any(rate_at_events .< 0)
        Turing.@addlogprob! -Inf
    else
        Turing.@addlogprob! sum(log.(rate_at_events)) - total_bg
    end

    lc = lc_cos_matrix*beta_cos .+ lc_sin_matrix*beta_sin

    (bg_segment=bg_segment, lc=lc, sigma_beta=sigma_beta) # beta_cos=beta_cos, beta_sin=beta_sin, 
end

@model function varying_background_spectral_fourier_model(pi_indices, segment_indices, segment_starts, segment_ends, cos_matrix, sin_matrix, lc_cos_matrix, lc_sin_matrix; period_amp_prior_frac=0.15)
    nseg = length(segment_starts)
    @assert length(segment_ends) == nseg

    ncounts, nfourier = size(cos_matrix)
    @assert size(sin_matrix) == (ncounts, nfourier)

    nchannels = maximum(pi_indices)

    T = sum(segment_ends .- segment_starts)
    mean_rate = ncounts/T

    mu_log_bg ~ Normal(log(mean_rate), 1)
    sigma_log_bg ~ Exponential(1)

    log_bg_segment ~ filldist(Normal(mu_log_bg, sigma_log_bg), nseg) 
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


end # module PulsarLightcurveExtraction
