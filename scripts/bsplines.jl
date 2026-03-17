## Imports
using BSplineKit
using CairoMakie
using FITSIO
using HDF5
using ProgressLogging

## Load events
event_time, event_phase, event_pi, segment_start, segment_stop = FITS(joinpath(@__DIR__, "..", "data", "J0740_merged_phase_0.25-3keV.fits.gz"), "r") do f
    event_time = read(f[2], "TIME")
    event_phase = read(f[2], "PULSE_PHASE")
    event_pi = read(f[2], "PI")

    segment_start = read(f[4], "START")
    segment_stop = read(f[4], "STOP")

    (event_time, event_phase, event_pi, segment_start, segment_stop)
end

segment_Ts = segment_stop .- segment_start;

## Load ARF
arf_e_high, arf_e_low, arf_e, arf_response, arf_start, arf_stop = h5open(joinpath(@__DIR__, "..", "data", "J0740_merged_arf.h5"), "r") do f
    map(x -> read(f, x), ("ENERG_HI", "ENERG_LO", "ENERGY", "SPECRESP", "START", "STOP"))
end

## Load RMF
rmf_e_high, rmf_e_low, rmf_response, rmf_start, rmf_stop = h5open(joinpath(@__DIR__, "..", "data", "J0740_merged_rmf.h5"), "r") do f
    rmf_e_high = read(f, "ENERG_HI")
    rmf_e_low = read(f, "ENERG_LO")
    rmf_response = read(f, "MATRIX")
    rmf_start = read(f, "START")
    rmf_stop = read(f, "STOP")

    return rmf_e_high, rmf_e_low, rmf_response, rmf_start, rmf_stop
end
rmf_e = 0.5*(rmf_e_high .+ rmf_e_low)


## Set up the B-spline basis
n_spec = 16
spec_order = 4
e_min = 0.2 # keV
e_max = 3.5 # keV
knots = exp.(range(log(e_min), log(e_max), length=n_spec-2))
basis = BSplineBasis(BSplineOrder(spec_order), knots)

## Set up the spline-to-energy matrix
spline_to_energy = zeros(Float32, length(rmf_e), length(basis))
for (i, e) in enumerate(rmf_e)
    j, vals = basis(e)
    if j >= spec_order
        spline_to_energy[i, j:-1:j-spec_order+1] .= vals
    end # Otherwise outside the range of the basis functions, so all zero.
end
spline_to_energy = reshape(rmf_e_high .- rmf_e_low, (:, 1)) .* spline_to_energy

## Combine rmf and arf 
combined_response = rmf_response .* reshape(arf_response, (1, size(arf_response)...))

## Put it all together
# Axes are PI, basis function index, times
# Want to sum over energy
spline_to_pi = stack([combined_response[:, :, t] * spline_to_energy for t in axes(combined_response, 3)], dims=3)

## Plot how the basis functions play out in energy space
f = Figure()
a = Axis(f[1,1], xlabel="PI", ylabel="Basis Functions", limits=((25, 300), (nothing, nothing)))
for j in axes(spline_to_pi, 2)
    lines!(a, spline_to_pi[:, j, 450], label="Basis Function $j")
end
# axislegend(a)
f