function errorbars_plotting(eigvals,Δeigvals,ylim_lower)
    errorbar_lower = similar(Δeigvals)
    for i in eachindex(Δeigvals)
        if eigvals[i] - Δeigvals[i] < 0
            errorbar_lower[i] = eigvals[i] - ylim_lower
        else
            errorbar_lower[i] = Δeigvals[i]
        end
    end
    return errorbar_lower
end
function plot_correlator!(plt,t,eigvals,Δeigvals;kws...)
    # remove negative entries 
    plot_eigvals = replace(x -> x < 0 ? NaN : x, eigvals)
    # but check if they are compatible with zeros
    #@assert minimum(eigvals + Δeigvals) > 0 
    # find smallest error bar that remains on the positive side 
    ylim_lower_v1 = minimum(filter(x -> x >0, eigvals - Δeigvals))  
    # alternatively, take smallest positive value and lower by one order of magnitude
    ylim_lower_v2 = minimum(filter(isfinite, plot_eigvals)) / 10
    ylim_lower = min(ylim_lower_v1, ylim_lower_v2)
    # set up error bars for plot
    errorbar_lower = errorbars_plotting(eigvals,Δeigvals,ylim_lower) 
    # add data to existing plot
    scatter!(plt,t,plot_eigvals,yerrors=(errorbar_lower, Δeigvals);kws...)
    return plt
end
function add_mass_band!(plt,m,Δm;label="",alpha=0.5,kws...)
    hspan!(plt,[m+Δm,m-Δm];label,alpha,kws...)
end
function add_fit_range!(plt,tmin,tmax,E,ΔE;label="",kws...)
    plot!(plt,tmin:tmax, E*ones(length(tmin:tmax)), ribbon = ΔE; label, kws...)
end