module LatticeUtils

    using NaNStatistics
    using Roots
    using Statistics
    using LinearAlgebra
    using Plots

    include("folding.jl")
    export correlator_folding
    include("effective_mass.jl")
    export implicit_meff, implicit_meff_jackknife, log_meff, log_meff_jackknife, asinh_meff, asinh_meff_jackknife, asinh_meff, asinh_meff_jackknife
    include("disconnected_loop_product.jl")
    export disconnected_loop_product
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
    include("errorstring.jl")
    export errorstring


end # module LatticeUtils
