using Bijectors, Turing, Enzyme, DynamicPPL, Mooncake

@model function model_tilde()
    x ~ LKJCholesky(8, 2.0)
end

@model function model_manual()
    x = Vector{Float64}(undef, 28)
    for i in eachindex(x)
        x[i] ~ Flat()
    end
    d = LKJCholesky(8, 2.0)
    b = bijector(d)
    bi = inverse(b)
    chol_factor = bi(x)
    Turing.@addlogprob! logpdf(d, chol_factor)
    Turing.@addlogprob! logabsdetjac(bi, x)
end

ads = [AutoEnzyme(mode=Enzyme.set_runtime_activity(Enzyme.Reverse)), AutoMooncake()]
params = randn(28)

ad_results = Dict()
for m in [model_tilde, model_manual]
    for ad in ads
        ad_results[m, ad] = DynamicPPL.TestUtils.AD.run_ad(m(), ad; benchmark=true, test=false, params=params) # Disable test; we will directly compare gradients below
    end
end

println("Grad diff for model_tilde: ", sum(abs.(1 .- ad_results[model_tilde, ads[1]].grad_actual ./ ad_results[model_tilde, ads[2]].grad_actual)))
println("Grad diff for model_manual: ", sum(abs.(1 .- ad_results[model_manual, ads[1]].grad_actual ./ ad_results[model_manual, ads[2]].grad_actual)))