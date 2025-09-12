using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using ArviZ
using CairoMakie
using Colors
using DataFrames
using DimensionalData
using Distributions
using FITSIO
using GaussianKDEs
using LaTeXStrings
using MCMCChainsStorage
using NCDatasets
using PairPlots
using Printf
using PulsarLightcurveExtraction
using Random
using Turing

phone_number = "J0740"
time_hdi = 4

f = FITS(joinpath(@__DIR__, "..", "data", "$(phone_number)_merged_phase_0.25-3keV.fits.gz"))
times = read(f[2], "TIME")
phases = read(f[2], "PULSE_PHASE")
pi_channel = read(f[2], "PI")
segment_starts = read(f[time_hdi], "START")
segment_ends = read(f[time_hdi], "STOP")

n_fourier = 10
si = segment_indices(times, segment_starts, segment_ends)
cm, sm = cos_sin_matrices(phases, n_fourier)

ph = 0.0:0.01:1.0
lc_cos_matrix, lc_sin_matrix = cos_sin_matrices(ph, n_fourier)

function husl_wheel(n)
    return [LCHuv(65, 90, h) for h in range(0, stop=360, length=n+1)][1:end-1]
end

function traceplot(chain; var_names=nothing)
    if var_names !== nothing
        p = chain.posterior[var_names]
    else
        p = chain.posterior
    end

    vars = keys(p)

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
            density!(adens, vec(zs_merge[merged_chain=At(c)]); label=nothing)
            lines!(atrace, vec(zs_merge[merged_chain=At(c)]); label=nothing)
        end
    end

    return f
end

chain = from_netcdf(joinpath(@__DIR__, "..", "data", "$(phone_number)_varying_background_fourier.nc"))

nseg = length(dims(chain.posterior, :segment))
sel = times .< segment_ends[nseg]

T_segment = segment_ends[1:nseg] .- segment_starts[1:nseg]
T = sum(segment_ends[1:nseg] .- segment_starts[1:nseg])
mean_rate = sum(sel)/T

sumry = ArviZ.summarize(chain)
min_ess = minimum(sumry.data.ess_bulk)
imin_ess = argmin(sumry.data.ess_bulk)
@info @sprintf("Min bulk ESS = %.1f for %s", min_ess, sumry.parameter_names[imin_ess])

traceplot(chain; var_names=(:mu_log_bg, :sigma_log_bg, :sigma_beta, :beta_cos, :beta_sin))

# A figure of merit: compare the sum of the inverse squared background rates in
# each segment to the sum of the inverse of the square of the average background
# counts.
mean_bg_segment = vec(mean(chain.posterior.bg_segment, dims=(:chain, :draw)))
mean_cos_amps = vec(mean(chain.posterior.beta_cos, dims=(:chain, :draw)))
mean_sin_amps = vec(mean(chain.posterior.beta_sin, dims=(:chain, :draw)))
mean_bg = sum(mean_bg_segment .* T_segment) / T

fom_segmented = sum(1 ./ (mean_bg_segment[si] .+ cm * mean_cos_amps .+ sm * mean_sin_amps).^2)
fom_average = sum(1 ./ (mean_bg .+ cm * mean_cos_amps .+ sm * mean_sin_amps).^2)
@sprintf("Figure of merit (segmented) = %.1f, (average) = %.1f, sqrt(ratio) = %.2f", fom_segmented, fom_average, sqrt(fom_average / fom_segmented))

f = Figure()
a = Axis(f[1,1], xlabel=L"b", palette=(patchcolor = husl_wheel(nseg),))                                  
for i in 1:nseg
    density!(a, vec(chain.posterior.bg_segment[segment=i]), label=nothing)
end
f
# save("/Users/wfarr/Downloads/j0030_bg.png", f)

f = Figure()
a = Axis(f[1,1], xlabel=L"\phi", ylabel=L"F(\phi)")
m = zeros(length(ph))
l = zeros(length(ph))
h = zeros(length(ph))
ll = zeros(length(ph))
hh = zeros(length(ph))
for i in eachindex(ph)
    lc = vec(chain.posterior.lc[phase=i])
    ll[i], l[i], m[i], h[i], hh[i] = quantile(lc, [0.025, 0.16, 0.5, 0.84, 0.975])
end
lines!(a, ph, m, label="Varying Background Model")
band!(a, ph, l, h, color=(Makie.wong_colors()[1], 0.25))
band!(a, ph, ll, hh, color=(Makie.wong_colors()[1], 0.25))
f
# save("/Users/wfarr/Downloads/lightcurve.png", f)

nf_start = 1
nf_end = n_fourier
d = Dict()
for i in nf_start:nf_end
    d["beta_cos_$(i)"] = vec(chain.posterior.beta_cos[fourier=At(i)])
    d["beta_sin_$(i)"] = vec(chain.posterior.beta_sin[fourier=At(i)])
end
df = DataFrame(d)
pairplot(df)

f = Figure()
a = Axis(f[1,1], xlabel="Fourier Mode", ylabel=L"\mu / \sigma", limits=(nothing, (-5, 5)))
sig_sin = Float64[]
sig_cos = Float64[]
fs = dims(chain.posterior, :fourier)
for ff in fs
    s = chain.posterior.beta_sin[fourier=At(ff)]
    c = chain.posterior.beta_cos[fourier=At(ff)]

    push!(sig_sin, mean(s)/std(s))
    push!(sig_cos, mean(c)/std(c))
end
fs_vec = vec(collect(fs))
lines!(a, fs_vec, sig_cos, label="cos")
lines!(a, fs_vec, sig_sin, label="sin")
band!(a, fs_vec, -1 .+ 0 .* fs_vec, 1 .+ 0 .* fs_vec, color=(:grey, 0.25))
band!(a, fs_vec, -2 .+ 0 .* fs_vec, 2 .+ 0 .* fs_vec, color=(:grey, 0.25))
band!(a, fs_vec, -3 .+ 0 .* fs_vec, 3 .+ 0 .* fs_vec, color=(:grey, 0.25))
axislegend(a)
f
# save("/Users/wfarr/Downloads/j0030_beta_significance.png", f)

## Binned model
nbins = 100
bin_bounds = range(0, stop=1, length=nbins+1)
bin_centers = 0.5*(bin_bounds[1:end-1] .+ bin_bounds[2:end])

bin_counts = zeros(nbins)
for p in phases
    bin_counts[searchsortedlast(bin_bounds, p)] += 1
end
bin_counts_uncertainty = sqrt.(bin_counts)

bin_fluxes = bin_counts ./ (T/nbins)
bin_flux_uncertainties = bin_counts_uncertainty ./ (T/nbins)

mean_a, AInvCholesky, M = binned_counts_fourier_model_mean_cholesky_precision(bin_centers, bin_fluxes, bin_flux_uncertainties, n_fourier)
mean_model = M * mean_a
M_nomean = copy(M)
M_nomean[:,1] .= 0.0

const_bg_lcs = zeros(nbins, 1000)
for i in 1:1000
    const_bg_lcs[:,i] = M_nomean * (mean_a .+ AInvCholesky.U \ randn(length(mean_a)))
end

m = zeros(nbins)
l = zeros(nbins)
ll = zeros(nbins)
h = zeros(nbins)
hh = zeros(nbins)
for i in 1:nbins
    m[i], l[i], h[i], ll[i], hh[i] = quantile(const_bg_lcs[i,:], [0.5, 0.16, 0.84, 0.025, 0.975])
end

f = Figure()
a = Axis(f[1,1], xlabel="Phase", ylabel="Count Rate", title="$(phone_number)")
lines!(a, bin_centers, m, label="Constant Background Model")
band!(a, bin_centers, l, h, color=(Makie.wong_colors()[2], 0.25))
band!(a, bin_centers, ll, hh, color=(Makie.wong_colors()[2], 0.25))
axislegend(a)
f

save("data/lightcurve.pdf", f)

f = Figure()
a = Axis(f[1,1], xlabel="Phase", ylabel="Count Rate", title="$(phone_number)")
hist!(a, phases, bins=250, weights=250/T*ones(length(phases)))
f
# save("/Users/wfarr/Downloads/j0030_lc_hist.png", f)

f = Figure()
a = Axis(f[1,1], xlabel="Phase", ylabel="Count Rate", title="$(phone_number)")
hist!(a, phases, bins=250, weights=250/T*ones(length(phases)), color=:grey)
lines!(a, bin_centers, m, label=nothing)
band!(a, bin_centers, l, h, color=(Makie.wong_colors()[1], 0.25))
band!(a, bin_centers, ll, hh, color=(Makie.wong_colors()[1], 0.25))
f
# save("/Users/wfarr/Downloads/j0030_lc_hist_with_fit.png", f)

f = Figure()
a = Axis(f[1,1], xlabel="Phase", ylabel="Count Rate", title="$(phone_number)")

for i in 1:nseg
    bg = mean(chain.posterior.bg_segment[segment=At(i)])
    hlines!(a, bg, color=(husl_wheel(nseg)[i], 0.1), label=nothing)
end

hist!(a, phases, bins=250, weights=250/T*ones(length(phases)), color=(:grey, 0.25))

lines!(a, bin_centers, m, label=nothing)
band!(a, bin_centers, l, h, color=(Makie.wong_colors()[1], 0.25))
band!(a, bin_centers, ll, hh, color=(Makie.wong_colors()[1], 0.25))
f
# save("/Users/wfarr/Downloads/j0030_lc_hist_with_fit_and_bg.png", f)

f = Figure()
a = Axis(f[1,1], xlabel=L"\phi", ylabel=L"F(\phi)")
m_b = zeros(length(ph))
l_b = zeros(length(ph))
h_b = zeros(length(ph))
ll_b = zeros(length(ph))
hh_b = zeros(length(ph))
for i in eachindex(ph)
    lc = vec(chain.posterior.lc[phase=i])
    ll_b[i], l_b[i], m_b[i], h_b[i], hh_b[i] = quantile(lc, [0.025, 0.16, 0.5, 0.84, 0.975])
end

no_bg_lcs = zeros(nbins, 1000)
for i in 1:1000
    no_bg_lcs[:,i] = M[:,2:end] * (mean_a[2:end] .+ (AInvCholesky.U \ randn(length(mean_a)))[2:end])
end

m_nb = zeros(nbins)
l_nb = zeros(nbins)
ll_nb = zeros(nbins)
h_nb = zeros(nbins)
hh_nb = zeros(nbins)
for i in 1:nbins
    m_nb[i], l_nb[i], h_nb[i], ll_nb[i], hh_nb[i] = quantile(no_bg_lcs[i,:], [0.5, 0.16, 0.84, 0.025, 0.975])
end

lines!(a, ph, zeros(length(ph)), label="Varying Background")
band!(a, ph, l_b .- m_b, h_b .- m_b, color=(Makie.wong_colors()[1], 0.1))
band!(ph, ll_b .- m_b, hh_b .- m_b, color=(Makie.wong_colors()[1], 0.1))

# m_b_interp = 0.5.*(m_b[1:end-1] .+ m_b[2:end])
# lines!(a, bin_centers, m_nb .- m_b_interp, label="Const Background")
# band!(a, bin_centers, l_nb .- m_b_interp, h_nb .- m_b_interp, color=(Makie.wong_colors()[2], 0.1))
# band!(a, bin_centers, ll_nb .- m_b_interp, hh_nb .- m_b_interp, color=(Makie.wong_colors()[2], 0.1))

lines!(a, bin_centers, zeros(length(bin_centers)), label="Const Background")
band!(a, bin_centers, l_nb .- m_nb, h_nb .- m_nb, color=(Makie.wong_colors()[2], 0.1))
band!(a, bin_centers, ll_nb .- m_nb, hh_nb .- m_nb, color=(Makie.wong_colors()[2], 0.1))

axislegend(a)
f
# save("/Users/wfarr/Downloads/lightcurve.png", f)

ibest = argmin(vec(median(chain.posterior.bg_segment, dims=(:chain, :draw))))
