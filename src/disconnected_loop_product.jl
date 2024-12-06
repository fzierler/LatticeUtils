# NOTE: Products of disconnected diagrams appear
#       For products of the same flavour I use the stochastic estimator of arXiv:1607.06654 eq. (14)
#       For products of different flavours we use a simple average over the sources. 
#       In both cases we need to take care of the relative time slices
function disconnected_loop_product(discon1,discon2;rescale=1,subtract_vev=false,nsrc_max=typemax(Int64))
    nconf1, nhits1, T = size(discon1)
    nconf2, nhits2, T = size(discon2)
    if nsrc_max < nhits1
        discon1 = discon1[:,1:nsrc_max,:]
        nhits1  = nsrc_max
    end
    if nsrc_max < nhits2
        discon2 = discon2[:,1:nsrc_max,:]
        nhits2  = nsrc_max
    end
    if subtract_vev
        vev1 = vev_contribution(discon1)
        vev2 = vev_contribution(discon2)
    end
    
    # permute dimensions for better memory acces
    discon1 = permutedims(discon1,(2,3,1))
    discon2 = permutedims(discon2,(2,3,1))
    
    nconf = min(nconf1,nconf2)
    timavg = zeros(eltype(discon1),(T,nconf))
    norm   = T*nhits1*nhits2
    
    if subtract_vev
        @inbounds for conf in 1:nconf, h in 1:nhits1, t in 1:T
            discon1[h,t,conf] = discon1[h,t,conf] - vev1[t]
        end
        @inbounds for conf in 1:nconf, h in 1:nhits2, t in 1:T
            discon2[h,t,conf] = discon2[h,t,conf] - vev2[t]
        end
    end
    
    sum_loop2 = sum(discon2,dims=1)
    for conf in 1:nconf
        for t in 1:T
            for t0 in 1:T
                Δt = mod(t-t0,T)
                @inbounds for hit1 in 1:nhits1
                    loop1 = discon1[hit1,t,conf]
                    timavg[Δt+1,conf] += sum_loop2[1,t0,conf]*loop1
                end
            end
        end
    end
    @. timavg = rescale*timavg/norm
    return permutedims(timavg,(2,1))
end
function disconnected_loop_product(discon;kws...)
    nconf, nhits, T = size(discon)
    disconnected_loop_product(discon[:,1:nhits÷2,:],discon[:,nhits÷2+1:nhits,:];kws...)
end
function vev_contribution(discon)
    vev = dropdims(mean(discon,dims=(1,2)),dims=(1,2)) 
    return vev
end