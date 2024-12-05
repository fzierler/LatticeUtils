# generate a resample of the original correlator matrix
function _resample_awi(corr)
    nconf,T = size(corr)
    samples = similar(corr)
    # temporary array for jackknife sampling
    tmp = zeros(eltype(corr),(nconf-1,T))
    for index in 1:nconf    
        for i in 1:index-1
            tmp[i,:] = corr[i,:]
        end
        for i in 1+index:nconf
            tmp[i-1,:] = corr[i,:]
        end
        # perform average after deleting one sample
        samples[index,:] = dropdims(mean(tmp,dims=1),dims=1)
    end
    return samples
end
function awi_corr(g0g5_g5,g5)
    # symmetrise the correlator
    AP = correlator_folding(g0g5_g5;t_dim=2,sign=-1)
    g5 = correlator_folding(g5;t_dim=2,sign=+1)
    
    dAP = correlator_derivative(AP;t_dim=2)
    awi_corr = dAP ./ g5 ./ 2
    return awi_corr
end
@. constfit(x,p) = p[1] + 0*x
function awi_fit(AWI::AbstractVector,ΔAWI::AbstractVector;tmin,tmax)
    T  = length(AWI)
    t0 = tmin:tmax
    # initial guess at T/2
    p0 = AWI[T÷2]
    fit  = curve_fit(constfit,t0,AWI[t0],ΔAWI[t0].^(-2),[p0])
    # extract mass and estimate error
    mAWI = fit.param[1]
    ΔmAWI = stderror(fit)[1]
    return mAWI, ΔmAWI
end
function awi_fit_jackknife(g0g5_g5::AbstractMatrix,g5::AbstractMatrix;tmin,tmax)
    awi = awi_corr(g0g5_g5,g5)
    awi_jackknife = _resample_awi(awi)
    N, T = size(awi_jackknife)
    mq_jackknife = zeros(N)
    for i in 1:N
        mq_jackknife[i] = first(awi_fit(awi_jackknife[i,:],ones(T);tmin,tmax))
    end
    return apply_jackknife(mq_jackknife)
end