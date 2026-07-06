# Post-mortem for a λ-collapsed checkpoint: replicate the _max_signal_k! formula
# on the host over every live cell of every level and dump the worst offenders
# with their full stored state.  Usage:  julia ckpt_smax_postmortem.jl <ckpt.jls>
using Serialization, Printf

ck = deserialize(ARGS[1])
γ = Float32(ck.gamma)
thr = length(ARGS) >= 2 ? parse(Float32, ARGS[2]) : 1.0f6
@printf("checkpoint: T=%s nsp=%d nstep=%s scheme=%s\n",
        string(ck.T), ck.nsp, string(ck.nstep[1:6]), string(ck.scheme))

for (lidx, ckl) in enumerate(ck.levels)
    l = lidx - 1
    nb = length(ckl.origins)
    nb == 0 && continue
    B3, _ = size(ckl.D)
    # per-level smax scan
    worst = Float32[]; wloc = Tuple{Int,Int}[]
    smax_lvl = 0.0f0
    nhuge = 0
    for i in 1:nb
        dsc = ckl.Dsc[i]; ssc = ckl.Ssc[i]; esc = ckl.Esc[i]
        for q in 1:B3
            ρraw = Float32(ckl.D[q, i]) * dsc
            # replicate the GATED _max_signal_k! (vacuum cells carry no signal)
            inv = ifelse(ρraw > 0.0f0, 1.0f0 / max(ρraw, 1.0f-30), 0.0f0)
            vx = Float32(ckl.S1[q, i]) * ssc * inv
            vy = Float32(ckl.S2[q, i]) * ssc * inv
            vz = Float32(ckl.S3[q, i]) * ssc * inv
            cs = sqrt(γ * (γ - 1.0f0) * max(Float32(ckl.Ge[q, i]) * esc, 1.0f-30) * inv)
            sm = max(abs(vx), max(abs(vy), abs(vz))) + cs
            smax_lvl = max(smax_lvl, sm)
            if sm > thr
                nhuge += 1
                if length(worst) < 8
                    push!(worst, sm); push!(wloc, (i, q))
                elseif sm > minimum(worst)
                    j = argmin(worst); worst[j] = sm; wloc[j] = (i, q)
                end
            end
        end
    end
    @printf("L%d: %6d blocks  smax=%.3e  cells>1e6: %d\n", l, nb, smax_lvl, nhuge)
    ord = sortperm(worst; rev = true)
    for j in ord
        (i, q) = wloc[j]
        dsc = ckl.Dsc[i]; ssc = ckl.Ssc[i]; esc = ckl.Esc[i]
        @printf("  block %5d cell %4d  sm=%.3e | D=%.4e S=(%.3e,%.3e,%.3e) Tau=%.4e Ge=%.4e | phys ρ=%.3e ρe=%.3e | sc=(%.1e,%.1e,%.1e) org=%s\n",
                i, q, worst[j],
                Float32(ckl.D[q, i]), Float32(ckl.S1[q, i]), Float32(ckl.S2[q, i]),
                Float32(ckl.S3[q, i]), Float32(ckl.Tau[q, i]), Float32(ckl.Ge[q, i]),
                Float32(ckl.D[q, i]) * dsc, Float32(ckl.Ge[q, i]) * esc,
                dsc, ssc, esc, string(Int.(ckl.origins[i])))
    end
end
