# scales.jl — per-block power-of-two f32 scale maintenance (the f16 phase).
#
# Convention: stored = physical / sc, sc = 2^e per block per class (D | S1–S3 |
# Tau+Ge), windowed so max|stored| ∈ [1,2).  Power-of-two scales make every
# rescale and every cross-block ghost conversion bit-exact (pure exponent shift).
# Rescaling runs BETWEEN root steps only (scales frozen inside a step) with a
# ±1-octave hysteresis so block scales don't thrash at window boundaries.

# one thread per live block: max |stored| over ALL nd³ cells (ghosts included —
# a subsequent fill must not overflow) per class.
@kernel function _blockmax_k!(maxD, maxS, maxE,
                              @Const(D), @Const(S1), @Const(S2), @Const(S3),
                              @Const(Tau), @Const(Ge), @Const(live_d),
                              nd::Int32, stride::Int32)
    t = @index(Global)
    @inbounds begin
        slot = live_d[Int32(t)]
        base = (slot - Int32(1)) * stride
        n3 = nd * nd * nd
        mD = 0.0f0; mS = 0.0f0; mE = 0.0f0
        for c in Int32(1):n3
            idx = base + c
            mD = max(mD, abs(Float32(D[idx])))
            mS = max(mS, max(abs(Float32(S1[idx])),
                             max(abs(Float32(S2[idx])), abs(Float32(S3[idx])))))
            mE = max(mE, max(abs(Float32(Tau[idx])), abs(Float32(Ge[idx]))))
        end
        maxD[slot] = mD; maxS[slot] = mS; maxE[slot] = mE
    end
end

# multiply one field of flagged blocks by a per-slot power-of-two ratio (exact)
@kernel function _rescale_k!(F, @Const(ratio), @Const(live_d),
                             nd::Int32, stride::Int32)
    t = @index(Global)
    t0 = Int32(t) - Int32(1)
    n3 = nd * nd * nd
    @inbounds begin
        slot = live_d[t0 ÷ n3 + Int32(1)]
        r = ratio[slot]
        if r != 1.0f0
            idx = (slot - Int32(1)) * stride + t0 % n3 + Int32(1)
            F[idx] = _narrow(eltype(F), Float32(F[idx]) * r)
        end
    end
end

"""
    update_scales!(hier, l; hyst = 1)

Re-window every live block's per-class scale: if max|stored| has drifted more
than `hyst` octaves out of [1,2), multiply the stored data by the exact
power-of-two ratio and fold it into the scale.  Call between root steps only.
"""
function update_scales!(hier::AMRHierarchy, l::Int; hyst::Int = 1)
    lev = hier.levels[l + 1]
    nb = length(lev.live)
    nb == 0 && return nothing
    maxD = device_zeros(lev.be, Float32, (lev.cap,))
    maxS = device_zeros(lev.be, Float32, (lev.cap,))
    maxE = device_zeros(lev.be, Float32, (lev.cap,))
    _blockmax_k!(lev.be)(maxD, maxS, maxE, gasfields(lev)..., lev.live_d,
                         Int32(lev.nd), Int32(lev.stride); ndrange = nb)
    hmx = (Array(maxD), Array(maxS), Array(maxE))
    scs = (lev.Dsc, lev.Ssc, lev.Esc)
    fieldsets = ((lev.D, lev.Do), (lev.S1, lev.S2, lev.S3, lev.S1o, lev.S2o, lev.S3o),
                 (lev.Tau, lev.Ge, lev.Tauo, lev.Geo))
    n = nb * lev.nd^3
    for cl in 1:3
        hsc = Array(scs[cl])
        ratio = ones(Float32, lev.cap)
        dirty = false
        for s in lev.live
            m = hmx[cl][s]
            (isfinite(m) && m > 0.0f0) || continue
            e = exponent(m)                       # m ∈ [2^e, 2^{e+1})
            if abs(e) > hyst
                ratio[s] = Float32(exp2(-e))      # restore window (exact)
                hsc[s]  *= Float32(exp2(e))
                dirty = true
            end
        end
        dirty || continue
        drat = to_device(lev.be, ratio, Float32)
        for f in fieldsets[cl]
            _rescale_k!(lev.be)(f, drat, lev.live_d, Int32(lev.nd),
                                Int32(lev.stride); ndrange = n)
        end
        copyto!(scs[cl], hsc)
    end
    return nothing
end

"""
    encode_from_host!(lev, hD, hS1, hS2, hS3, hTau, hGe)

Set a level's state from PHYSICAL host pools (Float64, pool layout): per live
block compute the per-class power-of-two scales (identity for Float32 storage),
encode stored = phys/sc, upload data + scales.  The IC/staging path for f16.
"""
function encode_from_host!(lev::Level, hD, hS1, hS2, hS3, hTau, hGe)
    T = eltype(lev.D)
    f16 = T === Float16
    n = lev.cap * lev.stride
    hin  = (hD, hS1, hS2, hS3, hTau, hGe)
    hout = ntuple(_ -> zeros(T, n), 6)
    cls  = (1, 2, 2, 2, 3, 3)                          # field → scale class
    hsc  = (ones(Float32, lev.cap), ones(Float32, lev.cap), ones(Float32, lev.cap))
    for s in lev.live
        base = (Int(s) - 1) * lev.stride
        rng = base+1:base+lev.nd^3
        if f16
            for cl in 1:3
                m = 0.0
                for fi in 1:6
                    cls[fi] == cl || continue
                    a = hin[fi]
                    @inbounds for idx in rng
                        m = max(m, abs(Float64(a[idx])))
                    end
                end
                m > 0 && (hsc[cl][s] = Float32(exp2(exponent(m))))
            end
        end
        for fi in 1:6
            a = hin[fi]; o = hout[fi]
            sc = Float64(hsc[cls[fi]][s])
            @inbounds for idx in rng
                o[idx] = _narrow(T, Float32(a[idx] / sc))
            end
        end
    end
    for (dev, o) in zip(gasfields(lev), hout)
        copyto!(dev, o)
    end
    copyto!(lev.Dsc, hsc[1]); copyto!(lev.Ssc, hsc[2]); copyto!(lev.Esc, hsc[3])
    return nothing
end
