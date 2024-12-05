module LatticeUtils

    using NaNStatistics
    using Roots
    using Statistics
    using LinearAlgebra

    include("folding.jl")
    export correlator_folding
    include("effective_mass.jl")
    export implicit_meff
    include("unbiased_estimator.jl")
    export unbiased_estimator
    include("variational_analysis.jl")
    export eigenvalues, eigenvalues_jackknife_samples, eigenvalues_eigenvectors, eigenvalues_eigenvectors_jackknife_samples
    include("plotting.jl")
    export add_mass_band!, add_fit_range!, plot_correlator!
    include("correlator_derivative.jl")
    export correlator_derivative
    include("fitcorr.jl")
    export fit_corr
    include("pcac.jl")
    export awi_corr, awi_fit


end # module LatticeUtils
