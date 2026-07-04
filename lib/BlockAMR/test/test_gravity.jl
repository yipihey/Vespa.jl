# test_gravity.jl — topgrid-potential gravity kick on blocks: the interpolated
# central-difference acceleration matches the same discrete operator evaluated
# on the host, on BOTH levels (exact split-position representation), f32 and f16.
const BA = BlockAMR

for BE in BACKENDS
@testset "topgrid-φ kick matches host operator [$BE]" begin
    for T in (Float32, Float16)
        hier = center_refined(; backend = BE, T)
        prof = x -> (1.0, 0.0, 0.0, 0.0, 1.0)
        set_ic!(hier, 0, prof); set_ic!(hier, 1, prof)
        nb = hier.nbase[1]
        # φ(z) = A cos(2πz): smooth, periodic
        A = 0.01
        hφ = zeros(Float32, nb^3)
        for k in 0:nb-1, j in 0:nb-1, i in 0:nb-1
            hφ[(k*nb+j)*nb+i+1] = Float32(A * cos(2π * (k + 0.5) / nb))
        end
        φ = BA.to_device(hier.be, hφ, Float32)
        dt2 = 1e-3
        for l in 0:1
            grav_kick_level!(hier, l, φ, dt2)
        end
        # host reference: same trilinear+CD operator at each cell center
        phi_tl(x, y, z) = begin
            xc = x*nb - 0.5; yc = y*nb - 0.5; zc = z*nb - 0.5
            i0 = floor(Int, xc); fx = xc - i0
            j0 = floor(Int, yc); fy = yc - j0
            k0 = floor(Int, zc); fz = zc - k0
            g(i, j, k) = hφ[(mod(k,nb)*nb + mod(j,nb))*nb + mod(i,nb) + 1]
            (1-fx)*((1-fy)*((1-fz)*g(i0,j0,k0)+fz*g(i0,j0,k0+1)) +
                    fy*((1-fz)*g(i0,j0+1,k0)+fz*g(i0,j0+1,k0+1))) +
            fx*((1-fy)*((1-fz)*g(i0+1,j0,k0)+fz*g(i0+1,j0,k0+1)) +
                fy*((1-fz)*g(i0+1,j0+1,k0)+fz*g(i0+1,j0+1,k0+1)))
        end
        h = 1.0 / nb
        worst = 0.0; amax = 0.0
        for l in 0:1
            lev = hier.levels[l+1]
            dx = BA.level_dx(hier, l)
            hS3 = Array(lev.S3); hsc = Array(lev.Ssc); hD = Array(lev.D); hdsc = Array(lev.Dsc)
            for s in lev.live
                m = lev.meta[s]; base = (Int(s)-1)*lev.stride
                for (ci, cj, ck) in ((0,0,0), (7,3,11), (15,15,15))
                    x = (Float64(m.origin[1]) + ci + 0.5) * dx
                    y = (Float64(m.origin[2]) + cj + 0.5) * dx
                    z = (Float64(m.origin[3]) + ck + 0.5) * dx
                    az = -(phi_tl(x, y, mod(z + h, 1.0)) - phi_tl(x, y, mod(z - h, 1.0))) / 2h
                    idx = base + ((ck+lev.ng)*lev.nd + (cj+lev.ng))*lev.nd + (ci+lev.ng) + 1
                    ρ = Float64(hD[idx]) * hdsc[s]
                    got = Float64(hS3[idx]) * hsc[s]
                    want = ρ * az * dt2
                    amax = max(amax, abs(want))
                    worst = max(worst, abs(got - want))
                end
            end
        end
        tol = T === Float16 ? 3e-3 * max(amax, 1e-12) : 1e-5 * max(amax, 1e-12)
        @test worst < max(tol, T === Float16 ? 1e-7 : 1e-11)
    end
end
end # for BE
