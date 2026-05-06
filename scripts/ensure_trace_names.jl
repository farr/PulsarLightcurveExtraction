#!/usr/bin/env julia
# Repair dimension names in an ArviZ trace NetCDF produced by the old
# from_mcmcchains call that was missing several variables from its `dims`
# dict.  Variables omitted from that dict get auto-generated dimension
# labels like `{varname}_dim_N`; this script renames them to the correct
# semantic labels and adds the corresponding coordinates.
#
# The file is edited in-place: data are written to a temp file alongside
# the original and then renamed over it.
#
# Usage:
#   julia scripts/ensure_trace_names.jl path/to/trace.nc

using ArgParse

let s = ArgParseSettings(description = "Fix dimension names in an ArviZ trace NetCDF file.")
    @add_arg_table! s begin
        "trace_path"
            help     = "Path to the trace NetCDF file to fix in place"
            required = true
    end
    global parsed_args = parse_args(s)
end

using ArviZ
using DimensionalData
using NCDatasets

inpath = parsed_args["trace_path"]

@info "Loading trace from $inpath …"
trace = from_netcdf(inpath)
post  = trace.posterior

# ── Borrow existing correctly-labelled dimension objects from the dataset ─
# This reuses the exact same Dim (with its Lookup) rather than rebuilding.
spec_dim = dims(post.mu_log_bg, Dim{:spec})
seg_dim  = dims(post.log_bg,    Dim{:segment})
n_spec   = length(spec_dim)
bi_dim   = (n_spec * (n_spec - 1)) ÷ 2

# These two are genuinely new (not yet in the dataset).
spec2_dim    = Dim{:spec2}(val(spec_dim))
cholesky_dim = Dim{:cholesky_corr_param}(1:bi_dim)

@info "Detected n_spec=$n_spec  n_seg=$(length(seg_dim))  bi_dim=$bi_dim"

# ── Build rename table ────────────────────────────────────────────────────
# ArviZ labels unlabelled variable-specific dimensions "{varname}_dim_N"
# where N counts from 1 (chain = implicit, draw = dim_0, first variable
# dimension = dim_1, second = dim_2, …).
#
# log_bg_raw is stored (spec, segment): dim_1 = n_spec, dim_2 = n_seg.
#
# cholesky_cov_log_bg and cov_log_bg are n_spec × n_spec; their two axes
# get distinct names :spec (dim_1) and :spec2 (dim_2).
dim_renames = Dict{Symbol,DimensionalData.Dimension}(
    :mu_log_bg_raw_dim_1                      => spec_dim,
    :log_fg_coeff_const_raw_dim_1             => spec_dim,
    :log_bg_raw_dim_1                         => spec_dim,
    :log_bg_raw_dim_2                         => seg_dim,
    :unconstrained_cholesky_corr_log_bg_dim_1 => cholesky_dim,
    :cholesky_cov_log_bg_dim_1                => spec_dim,
    :cholesky_cov_log_bg_dim_2                => spec2_dim,
    :cov_log_bg_dim_1                         => spec_dim,
    :cov_log_bg_dim_2                         => spec2_dim,
)

# ── Apply renames to every variable in the posterior ─────────────────────
function fix_dims(arr::AbstractDimArray, renames)
    new_dims = map(dims(arr)) do d
        get(renames, DimensionalData.name(d), d)
    end
    return rebuild(arr; dims = new_dims)
end

@info "Fixing posterior dimension names …"
var_names   = keys(post)                  # Tuple of Symbols
fixed_arrays = Tuple(fix_dims(post[k], dim_renames) for k in var_names)
new_post     = ArviZ.Dataset(NamedTuple{var_names}(fixed_arrays))

# ── Reconstruct InferenceData, carrying all other groups through unchanged ─
# parent(trace) is the NamedTuple of groups; merge replaces :posterior only.
new_trace = InferenceData(merge(parent(trace), (posterior = new_post,)))

# ── Write to a temp file then rename atomically over the original ─────────
tmppath = inpath * ".tmp.nc"
@info "Writing fixed trace to $tmppath …"
to_netcdf(new_trace, tmppath)

@info "Replacing original: $tmppath → $inpath"
mv(tmppath, inpath; force = true)

@info "Done."
