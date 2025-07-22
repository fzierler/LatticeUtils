function _bin_correlator_matrix(corr;binsize=2)
    nop, nop, N, T = size(corr)
    corr_binned = zeros(eltype(corr), nop, nop, N÷binsize, T)
    for i in 1:N÷binsize
        for j in 1:binsize
            offset = (i-1)*binsize
            corr_binned[:,:,i,:] += corr[:,:,offset+j,:]/binsize
        end 
    end
    return corr_binned
end
eigenvalues(corr;kws...) = first(eigenvalues_eigenvectors(corr;kws...))
eigenvalues_jackknife_samples(corr;kws...) = first(eigenvalues_eigenvectors_jackknife_samples(corr;kws...))
function eigenvalues_eigenvectors(corr;kws...)
    eigvals_jk, eigvecs_jk = eigenvalues_eigenvectors_jackknife_samples(corr;kws...)
    eigvals, Δeigvals = apply_jackknife(eigvals_jk;dims=2)
    eigvecs, Δeigvecs = apply_jackknife(eigvecs_jk;dims=3)
    return eigvals, Δeigvals, eigvecs, Δeigvecs
end
function eigenvalues_eigenvectors_jackknife_samples(corr;kws...)
    sample = delete1_resample(corr)
    eigenvalues_eigenvectors_from_samples(sample;kws...)
end
function eigenvalues_eigenvectors_from_samples(sample;t0,gevp=true,sortby=x-> abs(x))
    nops, nconf, T = size(sample)[2:4]
    # Always save the results as 64bit floating point number even when we are using higher
    # precision datatypes for solivng the (generalised) eigenvalue problem
    eigvals_jk = zeros(Float64,(nops,nconf,T))
    eigvecs_jk = zeros(ComplexF64,(nops,nops,nconf,T))
    for s in 1:nconf, t in 1:T
        if gevp
            t1  = t < T÷2 + 1 ? t0 : T - t0 + 2
            try
                Ct  = Hermitian(sample[:,:,s,t])
                Ct0 = Hermitian(sample[:,:,s,t1])
                sol = eigen(Ct,Ct0,sortby=sortby)
            catch
                Ct  = sample[:,:,s,t]
                Ct0 = sample[:,:,s,t1]
                sol = eigen(Ct,Ct0,sortby=sortby)
            end
        else
            sol = eigen(Hermitian(sample[:,:,s,t]),sortby=sortby)
        end
        eigvals_jk[:,s,t] = Float64.(real.(sol.values))
        for i in 1:nops
            eigvecs_jk[:,i,s,t] = normalize(sol.vectors[:,i])
        end
    end
    return eigvals_jk, eigvecs_jk
end

# generate a resample of the original correlator matrix
function delete1_resample(corr_matrix)
    nops,nconf,T = size(corr_matrix)[2:end]
    samples = similar(corr_matrix)
    # Strategy: Sum over all configs then remove one
    tmp = dropdims(sum(corr_matrix,dims=3),dims=3)
    for index in 1:nconf    
        @. samples[:,:,index,:] = (tmp - corr_matrix[:,:,index,:])/(nconf-1) 
    end
    return samples
end
# apply jackknife resampling along dimension dims
function apply_jackknife(obs::AbstractArray;dims::Integer)
    N  = size(obs)[dims]
    O  = dropdims(mean(obs;dims);dims)
    ΔO = dropdims(sqrt(N-1)*std(obs;dims,corrected=false);dims)
    return O, ΔO
end
function apply_jackknife(obs::AbstractVector)
    N  = length(obs)
    O  = mean(obs)
    ΔO = sqrt(N-1)*std(obs,corrected=false)
    return O, ΔO
end
# apply jackknife while ignoring NaNs
function nan_apply_jackknife(obs::AbstractArray;dims::Integer)
    N  = size(obs)[dims]
    O  = dropdims(nanmean(obs;dims);dims)
    ΔO = dropdims(sqrt(N-1)*nanstd(obs;dims,corrected=false);dims)
    return O, ΔO
end
function nan_apply_jackknife(obs::AbstractVector)
    N  = length(obs)
    O  = nanmean(obs)
    ΔO = sqrt(N-1)*nanstd(obs,corrected=false)
    return O, ΔO
end
function cov_jackknife_eigenvalues(evjk::AbstractArray)
    Nev, Nsamples, T = size(evjk) 
    covm = zeros(Nev,T,T)
    for N in 1:Nev
        c0 = (Nsamples-1)*cov(evjk[N,:,:],dims=1,corrected=false)
        covm[N,:,:] = Hermitian(c0)
    end
    return covm
end