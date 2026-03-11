## Imports
using ArviZ
using CairoMakie
using DataFrames
using Distributions
using Mooncake
using PairPlots
using PulsarLightcurveExtraction
using StatsFuns
using Turing

## Set up the data
N = 100
log_bg = randn(N)

log_fg = randn()
log_fg_proj = randn(N)

log_total = logaddexp.(log_bg, log_fg_proj .+ log_fg)
log_total_obs = log_total .+ randn(N)

## Model
@model function model(log_total_obs, log_fg_proj)
    mu_log_bg ~ Normal(0, 1)
    sigma_log_bg ~ Exponential(1)

    log_fg ~ Normal(0, 1)

    log_bg_raw = Vector{Float64}(undef, length(log_total_obs))
    log_bg = Vector{Float64}(undef, length(log_total_obs))

    for i in eachindex(log_bg)
        log_bg_raw[i] ~ Normal(0, 1)
        log_bg[i] := mu_log_bg + sigma_log_bg * log_bg_raw[i]
    end

    for i in eachindex(log_total_obs)
        log_total_obs[i] ~ Normal(logaddexp(log_bg[i], log_fg_proj[i] + log_fg), 1.0)
    end
end

## Sample it
chain = sample(model(log_total_obs, log_fg_proj), NUTS(1000, 0.8; adtype=AutoMooncake()), 1000)

## Convert to InferenceData
trace = from_mcmcchains(chain; coords=Dict(:log_bg => (:obs,)), dims=Dict(:obs => 1:length(log_total_obs)))

## ESS obvious
minimum(ess(trace))

## Plot it obvious
PulsarLightcurveExtraction.traceplot(trace)

## Parameter Plot
df = DataFrame(Dict(
    :log_fg => vec(trace.posterior.log_fg),
    :mu_log_bg => vec(trace.posterior.mu_log_bg),
    :sigma_log_bg => vec(trace.posterior.sigma_log_bg)
))

pairplot(df, PairPlots.Truth((; :log_fg => log_fg, :mu_log_bg => 0.0, :sigma_log_bg => 1.0)))
