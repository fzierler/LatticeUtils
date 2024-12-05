# see equation (10) in arXiv:1607.06654 [hep-lat]
# (Notation in Gattringer/Lang is misleading!)
function _meff_at_t(c::AbstractVector,t,T;sign=+1)
    # make an overall minus sign explicit 
    # removeing the absolute value abs() below leads to frequent
    # errors due to the behaviour at small t in conjuction with
    # numerical derivatives.
    s = t <= T/2 ? +1 : -1
    # non-implicit mass as initial guess
    m0 = log(abs(c[mod1(t+1,T)]/c[t]))
    t0 = mod1(t+1,T)
    # correlator at large times (dropped overall factor)
    cor_lt(m,T,t) = exp(-m*t) + sign*exp(-m*(T-t))
    # function to fit the effective mass
    g(m,T,t,t0) = cor_lt(m,T,t0)/cor_lt(m,T,t) - c[t0]/c[t] 
    # Use the more simpler algorithms from the Roots.jl package
    # find_zero() has more overhead and fails if the algorithm does not converged
    # Here we just use two simple, derivative free methods. If they do not converge
    # they return NaN. If that is the case then we try a slightly more robust algorithm.
    m = Roots.secant_method(x->g(x,T,t,t0),m0;maxevals=10000)
    if isnan(m)
       m = Roots.dfree(x->g(x,T,t,t0),m0)
    end
    # multiply by overall sign that has been previously extracted
    return abs(m)
end
function meff_from_jackknife(eigenvalues_jk;sign=+1,swap=nothing)
    nops, nsamples, T = size(eigenvalues_jk)
    meff_jk = similar(eigenvalues_jk)
    for op in 1:nops, sample in 1:nsamples
        meff_jk[op,sample,:] = implicit_meff(eigenvalues_jk[op,sample,:];sign) 
    end
    meff, Δmeff = nan_apply_jackknife(meff_jk;dims=2)
    if !isnothing(swap)
        _swap_at_crossing(meff, swap)
        _swap_at_crossing(Δmeff,swap)    
    end
    return meff, Δmeff
end
"""
    implicit_meff(c::AbstractVector;sign=+1)

Calculates the effective mass of the correlator `c` according to equation (10)
of arXiv:1607.06654. By default, a periodic correlator is assumed. For an 
anti-periodic correlator choose `sign=-1`.

    implicit_meff(c::AbstractVector,Δc::AbstractVector;sign=+1)

Includes a crude estimate of the uncertainty of the effective mass, when the 
standard uncertainty `Δc` is provided. For a more reliable provide an array 
containing measurements of the correlator on each configuration. This will leads
to a more reliable estimation using the jackknife method.

    implicit_meff(c::AbstractArray;sign=+1)

Calculates the effective mass of the correlator `c` according to equation (10)
of arXiv:1607.06654 and provides an estimator of the uncertainty using a 
jackknife analysis. 

By default, a periodic correlator is assumed. For an  anti-periodic correlator 
choose `sign=-1`. The data `c` is assumed to be an array where the first index 
corresponds to the Euclidean time and the second one to the Monte-Carlo samples. 
"""
function implicit_meff(c::AbstractVector;sign=+1)
    T = length(c)
    m = similar(c)
    for t in 1:T
        m[t] = _meff_at_t(c,t,T;sign)
    end
    return m
end
function implicit_meff(c::AbstractVector,Δc::AbstractVector;sign=+1)
    m1 = implicit_meff(c;sign)
    m2 = implicit_meff(c + Δc;sign)
    m3 = implicit_meff(c - Δc;sign)
    Δm = @. abs(m3-m2)/2
    return m1, Δm
end
function implicit_meff(corrs::AbstractMatrix;sign=+1)
    T, N = size(corrs)
    # create arrays for decay constant
    corrs_delete1 = zeros(T,N-1)
    meff = zeros(N,T)
    # set up jack-knife (one deletion)
    for i in 1:N
        for t in 1:T
            for j in 1:N
               (j < i) && (corrs_delete1[t,j]   = corrs[t,j])
               (j > i) && (corrs_delete1[t,j-1] = corrs[t,j])
            end
        end
        # perform averaging for fitting weights
        C = reshape(mean(corrs_delete1,dims=2),T)
        meff[i,:] = implicit_meff(C;sign)
    end
    return nan_apply_jackknife(meff,dims=1)
end
# This function takes care of arrays with additional inidces. The additional 
# indices are collected as ind by the use of the `axes(corr)` function. This 
# returns a set of iterators for each additional dimension of the array. We then
# loop over every additional index using the `Iterators.product` utility 
function implicit_meff(corrs::AbstractArray;sign=+1)
    d = ndims(corrs)
    it, iMC, ind... = axes(corrs)
    size_meff_array = (size(corrs,1),size(corrs)[3:end]...)
    meff  = zeros(eltype(corrs),size_meff_array)
    Δmeff = zeros(eltype(corrs),size_meff_array)
    #@showprogress "Jackknife analysis for effective mass:" for i in Iterators.product(ind...)
    for i in Iterators.product(ind...)
        # slurping is needed to correctly insert the tuple i into an index
        c = @view corrs[:,:,i...]
        meff[:,i...], Δmeff[:,i...] = implicit_meff_jackknife(c;sign)
    end
    return meff, Δmeff
end
