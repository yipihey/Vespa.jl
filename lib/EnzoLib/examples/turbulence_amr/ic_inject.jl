# Shared: inject a uniform-density solenoidal turbulence IC into the live Enzo top
# grid (overwrites Density / Velocity1-3 / TotalEnergy / InternalEnergy). Same
# construction as the PPMKernels bench: Σ_k A_k ê⊥(k) cos(2π k·x+φ), A_k ∝
# |k|^(-specidx/2), normalized to the target RMS Mach over the active interior.
# Field positional order is [0=Density,1=Vel1,2=Vel2,3=Vel3,4=TotalEnergy,5=GasEnergy].

function inject_turbulence!(h; mach, gamma, ng, cs = 1.0, seed = 271, kmin = 2, kmax = 3, specidx = 4.0)
    nx, ny, nz = EnzoLib.problem_grid_dims(h, 0); N = nx * ny * nz
    nax = nx - 2ng; dxn = 1.0 / nax
    X(i) = (i - ng + 0.5) * dxn                                 # periodic cell-centre coord
    Random.seed!(seed)
    vx = zeros(N); vy = zeros(N); vz = zeros(N)
    modes = [(kx, ky, kz) for kx in -kmax:kmax, ky in -kmax:kmax, kz in -kmax:kmax
             if kmin^2 <= kx^2 + ky^2 + kz^2 <= kmax^2]
    for (kx, ky, kz) in modes
        kk = sqrt(kx^2 + ky^2 + kz^2); amp = kk^(-specidx / 2); kh = (kx, ky, kz) ./ kk
        a = randn(3); a .-= dot(a, collect(kh)) .* collect(kh); na = norm(a)
        na < 1e-12 && continue; a ./= na
        φ = 2π * rand(); a1, a2, a3 = amp .* a
        @inbounds for k in 0:nz-1, j in 0:ny-1, i in 0:nx-1
            s = cos(2π * (kx * X(i) + ky * X(j) + kz * X(k)) + φ)
            q = i + nx * j + nx * ny * k + 1
            vx[q] += a1 * s; vy[q] += a2 * s; vz[q] += a3 * s
        end
    end
    s2 = 0.0; nc = 0
    @inbounds for k in ng:nz-ng-1, j in ng:ny-ng-1, i in ng:nx-ng-1
        q = i + nx * j + nx * ny * k + 1; s2 += vx[q]^2 + vy[q]^2 + vz[q]^2; nc += 1
    end
    f = mach * cs / sqrt(s2 / nc); vx .*= f; vy .*= f; vz .*= f
    eint0 = (cs^2 / gamma) / (gamma - 1)
    D = ones(N); TE = eint0 .+ 0.5 .* (vx .^ 2 .+ vy .^ 2 .+ vz .^ 2); IE = fill(eint0, N)
    EnzoLib.problem_set_field(h, 0, D); EnzoLib.problem_set_field(h, 1, vx)
    EnzoLib.problem_set_field(h, 2, vy); EnzoLib.problem_set_field(h, 3, vz)
    EnzoLib.problem_set_field(h, 4, TE); EnzoLib.problem_set_field(h, 5, IE)
    return nothing
end
