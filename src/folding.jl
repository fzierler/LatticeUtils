"""
    correlator_folding(c::AbstractArray;t_dim=1,sign=+1)

Fold correlator along the center of the Euclidean time extent.
By default a symmetric correlator is assumed and a relatice sign of +1 is used.
For anti-symmetric correlators use `sign=-1`. 

By default, the temporal dimension of the array containing the correlator data 
`c` is assumed to be in the 1st dimension. This can be changed by providing an 
explicit dimension `t_dim` as a keyword argument
"""
function correlator_folding(c::AbstractArray;t_dim=1,sign=+1)
    c_folded = similar(c)
    # Get the number of Euclidean timeslices in the specified dimension
    T = size(c)[t_dim]
    # get the overall number of dimensions for contructing a unit vector in the
    # temporal direction.
    n = ndims(c)
    δt(i) = ifelse(i == t_dim,1,0)
    # ntuple applies the function in the first argument δt to every index `1:n`
    # Thus, this corresponds to a delta-step in the temporal direction, i.e. a 
    # unit vector in Euclidean time. 
    unit_t = CartesianIndex(ntuple(δt,n))
    # Use julia's Cartesian indices to index arrays of arbitrary dimension 
    for i in CartesianIndices(c)
        t = i[t_dim]
        # Only consider first half of the temproal direction since the 
        # second half is going to be used in the folding anyway
        t > T ÷ 2 && continue
        # Add 1 to all indices: julia is one-indexed
        # This skips the first entry but it does not to be averaged anyhow
        t1 = t+1
        t2 = T-t+1
        # construct the corresponding indices the are used in folding
        ind1 = i + (t1-t)*unit_t
        ind2 = i + (t2-t)*unit_t
        # fold the correlator: choose the sign based on the symmetry/anti-symmetry
        # of the correlator
        c_folded[ind1] = (c[ind1] + sign*c[ind2])/2
        c_folded[ind2] = (sign*c[ind1] + c[ind2])/2
        # take care of the first time index which is not folded at all
        if t == 1 
            c_folded[i] = c[i]
        end
    end
    return c_folded
end