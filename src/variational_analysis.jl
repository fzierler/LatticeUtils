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
function eigenvalues(corr;t0=1)
    eigvals_jk = eigenvalues_jackknife_samples(corr;t0)
    eigvals, Δeigvals = apply_jackknife(eigvals_jk;dims=2)
    return eigvals, Δeigvals
end
function eigenvalues_eigenvectors(corr;t0=1)
    eigvals_jk, eigvecs_jk = eigenvalues_eigenvectors_jackknife_samples(corr;t0)
    eigvals, Δeigvals = apply_jackknife(eigvals_jk;dims=2)
    eigvecs, Δeigvecs = apply_jackknife(eigvecs_jk;dims=3)
    return eigvals, Δeigvals, eigvecs, Δeigvecs
end
function eigenvalues_jackknife_samples(corr;t0 = 1, imag_thresh = 1E-11)
    sample = delete1_resample(corr)
    nops, nconf, T = size(sample)[2:4]
    eigvals_jk = zeros(eltype(sample),(nops,nconf,T))
    max_imag = 0.0
    for s in 1:nconf, t in 1:T
        # smaller values correspond to a faster decay, and thus correspond to a larger masses
        # use sortby to sort the eigenvalues by ascending eigen-energy of the meson state
        vals = eigen(sample[:,:,s,t],sample[:,:,s,t0]).values
        max_imag = max(max_imag, maximum(abs.(imag.(vals))))
        eigvals_jk[:,s,t] = real.(vals)
    end
    max_imag > imag_thresh && @warn "imaginary part of $max_imag exceeds threshold of $imag_thresh"
    return eigvals_jk
end
function eigenvalues_eigenvectors_jackknife_samples(corr;t0 = 1, imag_thresh = 1E-11)
    sample = delete1_resample(corr)
    nops, nconf, T = size(sample)[2:4]
    eigvals_jk = zeros(eltype(sample),(nops,nconf,T))
    eigvecs_jk = zeros(ComplexF64,(nops,nops,nconf,T))
    max_imag = 0.0
    for s in 1:nconf, t in 1:T
        # smaller values correspond to a faster decay, and thus correspond to a larger masses
        # use sortby to sort the eigenvalues by ascending eigen-energy of the meson state
        sol = eigen(sample[:,:,s,t],sample[:,:,s,t0],sortby= x-> abs(x))
        max_imag = max(max_imag, maximum(abs.(imag.(sol.values))))
        eigvals_jk[:,s,t] = real.(sol.values)
        # I am unsure if the average over all eigenvectors is correct.
        for i in 1:2
            eigvecs_jk[:,i,s,t] = normalize(sol.vectors[:,i])
        end
    end
    max_imag > imag_thresh && @warn "imaginary part of $max_imag exceeds threshold of $imag_thresh"
    return eigvals_jk, eigvecs_jk
end
# generate a resample of the original correlator matrix
function delete1_resample(corr_matrix)
    nops,nconf,T = size(corr_matrix)[2:end]
    samples = similar(corr_matrix)
    # temporary array for jackknife sampling
    tmp = zeros(eltype(corr_matrix),(nops,nops,nconf-1,T))
    for index in 1:nconf    
        for i in 1:index-1
            tmp[:,:,i,:] = corr_matrix[:,:,i,:]
        end
        for i in 1+index:nconf
            tmp[:,:,i-1,:] = corr_matrix[:,:,i,:]
        end
        # perform average after deleting one sample
        samples[:,:,index,:] = dropdims(mean(tmp,dims=3),dims=3)
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