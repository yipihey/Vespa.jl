# MUSCL (PLM + HLL) 1-D flux line certified against the LIVE Enzo hydro_rk solver.
#
# The reference is EnzoLib.hydro_rk_line → enzomodules_hydro_rk_line (the grid
# dylib's HydroLine: Enzo's actual PLM reconstruction + HLL Riemann flux). The KA
# port is diffed on the A/B/C ladder (A: cpu-f64 vs Enzo-f64, B: cpu-f32≡metal-f32,
# C: metal-f32 vs Enzo-f64). Requires the grid dylib (build_grid_darwin.sh).

using EnzoLib

@testset "MUSCL — PLM+HLL flux line vs live Enzo hydro_rk" begin
    if !EnzoLib.grid_available()
        @test_skip "grid dylib not built — skipping live hydro_rk MUSCL certification"
    else
        ncells, nghost = 24, 3
        active = ncells - 2 * nghost
        gamma, theta = 1.4, 1.5

        # primitive line: a shock + smooth structure (exercises the minmod limiter)
        rho = zeros(ncells); eint = zeros(ncells); vx = zeros(ncells)
        vy = zeros(ncells); vz = zeros(ncells)
        for i in 1:ncells
            t = tanh((i - 12.5) * 0.9)
            rho[i]  = 0.5625 - 0.4375 * t
            pr      = 0.5500 - 0.4500 * t
            vx[i]   = 0.20  - 0.15 * t
            vy[i]   = 0.10 + 0.03 * sinpi(i / 7)
            vz[i]   = -0.05
            eint[i] = pr / ((gamma - 1) * rho[i])
        end
        prim = permutedims(hcat(rho, eint, vx, vy, vz))   # 5×ncells (rows ρ,eint,vx,vy,vz)

        ref = EnzoLib.hydro_rk_line(prim; riemann = 1, gamma = gamma, theta = theta,
                                    nghost = nghost, small_rho = 1e-10, small_p = 1e-10)
        @test size(ref) == (5, active + 1)
        @test any(!iszero, ref)                            # non-vacuous

        run_mu(name, ::Type{T}, field::Int) where {T} = begin
            be = PPMKernels.backend(name)
            dev(a) = PPMKernels.to_device(be, a, T)
            z() = PPMKernels.device_zeros(be, T, (active + 1,))
            fd, fs1, fs2, fs3, fe = z(), z(), z(), z(), z()
            PPMKernels.muscl_flux_line!(fd, fs1, fs2, fs3, fe, dev(rho), dev(eint),
                                        dev(vx), dev(vy), dev(vz); ncells = ncells,
                                        nghost = nghost, gamma = gamma, theta = theta,
                                        small_rho = 1e-10)
            PPMKernels.to_host((fd, fs1, fs2, fs3, fe)[field])
        end

        for (field, name) in ((1, "density"), (2, "x-mom"), (3, "y-mom"), (4, "z-mom"), (5, "energy"))
            reff = ref[field, :]
            layerA!("muscl.$name", run_mu(:cpu, Float64, field), reff)
            layerB!("muscl.$name", (nm, T) -> run_mu(nm, T, field))
            layerC!("muscl.$name", (nm, T) -> run_mu(nm, T, field), reff)
        end
    end
end
