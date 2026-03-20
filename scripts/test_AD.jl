using Mooncake, LogDensityProblems, DifferentiationInterface, DynamicPPL
import DifferentiationInterface as DI

# model = your_model(data...)
vi = DynamicPPL.VarInfo(model)
ldf = DynamicPPL.LogDensityFunction(model) # , vi, DynamicPPL.SamplingContext())

θ = randn(LogDensityProblems.dimension(ldf))

# Mooncake gradient
mooncake_backend = AutoMooncake(; config=nothing)
mooncake_prep = DI.prepare_gradient(ldf, mooncake_backend, θ)
_, g_mooncake = DI.value_and_gradient(ldf, mooncake_backend, θ, mooncake_prep)

# Enzyme gradient  
enzyme_backend = AutoEnzyme(mode=Enzyme.set_runtime_activity(Enzyme.Reverse))
enzyme_prep = DI.prepare_gradient(ldf, enzyme_backend, θ)
_, g_enzyme = DI.value_and_gradient(ldf, enzyme_backend, θ, enzyme_prep)

println("Max abs error:  ", maximum(abs, g_enzyme .- g_mooncake))
println("Max rel error:  ", maximum(abs.(g_enzyme .- g_mooncake) ./ (abs.(g_mooncake) .+ 1e-10)))
println("Correlation:    ", cor(g_enzyme, g_mooncake))
println("Signs agree:    ", mean(sign.(g_enzyme) .== sign.(g_mooncake)))

# If there are disagreements, find which parameters are worst
worst = sortperm(abs.(g_enzyme .- g_mooncake), rev=true)
println("\nTop 10 worst parameter indices:")
for i in worst[1:10]
    println("  idx $i: enzyme=$(g_enzyme[i]),  mooncake=$(g_mooncake[i])")
end