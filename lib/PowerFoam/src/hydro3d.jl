# AREPO-shaped 3-D Euler finite-volume kernels.
#
# This is the 3-D analogue of hydro2d.jl: a face kernel computes one ALE flux
# per face, and a cell kernel gathers through a CSR cell-face table.  The first
# mesh producer is a periodic Cartesian Voronoi-equivalent face table; AREPO's
# exported 3-D Voronoi faces can feed the same array layout once wired in.

struct ArepoMeshArrays3D{I<:AbstractVector,R<:AbstractVector,S<:AbstractVector}
    c1::I
    c2::I
    cell_face_offsets::I
    cell_faces::I
    cell_face_signs::S
    volume::R
    face_area::R
    normal_x::R
    normal_y::R
    normal_z::R
    face_vx::R
    face_vy::R
    face_vz::R
end

struct EulerState3D{A<:AbstractVector}
    D::A
    Mx::A
    My::A
    Mz::A
    E::A
end

struct PrimitiveState3D{A<:AbstractVector}
    rho::A
    vx::A
    vy::A
    vz::A
    pressure::A
end

struct FaceFluxWork3D{A<:AbstractVector}
    FD::A
    FMx::A
    FMy::A
    FMz::A
    FE::A
end

struct FaceStates3D{A<:AbstractVector}
    left::A
    right::A
end

@inline _cell_id_3d(i, j, k, n) = i + n * (j - 1) + n * n * (k - 1)
@inline _wrap1(i, n) = i > n ? 1 : i

function _cell_face_csr(ncells::Integer, c1, c2, ::Type{I}) where {I<:Integer}
    counts = zeros(Int, ncells)
    @inbounds for f in eachindex(c1)
        counts[Int(c1[f])] += 1
        j = Int(c2[f])
        j > 0 && (counts[j] += 1)
    end
    offsets = Vector{I}(undef, ncells + 1)
    offsets[1] = one(I)
    @inbounds for i in 1:ncells
        offsets[i + 1] = offsets[i] + I(counts[i])
    end
    faces = Vector{I}(undef, Int(offsets[end] - one(I)))
    signs = Vector{I}(undef, length(faces))
    cursor = Int.(offsets[1:end-1])
    @inbounds for f in eachindex(c1)
        i = Int(c1[f])
        p = cursor[i]
        faces[p] = I(f)
        signs[p] = -one(I)
        cursor[i] += 1
        j = Int(c2[f])
        if j > 0
            p = cursor[j]
            faces[p] = I(f)
            signs[p] = one(I)
            cursor[j] += 1
        end
    end
    return offsets, faces, signs
end

function _velocity_matrix3(cell_velocity, n::Integer)
    if cell_velocity === nothing
        return nothing
    elseif cell_velocity isa AbstractMatrix
        size(cell_velocity) == (n, 3) || error("cell_velocity matrix must be n x 3")
        return Matrix{Float64}(cell_velocity)
    else
        length(cell_velocity) == n || error("cell_velocity length must match cell count")
        out = Matrix{Float64}(undef, n, 3)
        for i in 1:n
            v = cell_velocity[i]
            out[i, 1] = float(v[1])
            out[i, 2] = float(v[2])
            out[i, 3] = float(v[3])
        end
        return out
    end
end

function _face_velocity_arrays_3d(c1, c2, face_velocity, cell_velocity,
                                  ::Type{T}) where {T<:AbstractFloat}
    nf = length(c1)
    vx = zeros(T, nf)
    vy = zeros(T, nf)
    vz = zeros(T, nf)
    if face_velocity !== nothing
        if face_velocity isa AbstractMatrix
            size(face_velocity) == (nf, 3) || error("face_velocity must be nf x 3")
            vx .= T.(@view face_velocity[:, 1])
            vy .= T.(@view face_velocity[:, 2])
            vz .= T.(@view face_velocity[:, 3])
        else
            length(face_velocity) == nf || error("face_velocity length must match face count")
            for f in 1:nf
                v = face_velocity[f]
                vx[f] = T(v[1])
                vy[f] = T(v[2])
                vz[f] = T(v[3])
            end
        end
    elseif cell_velocity !== nothing
        n = maximum(Int.(c1))
        any(>(0), c2) && (n = max(n, maximum(Int.(filter(>(0), c2)))))
        cm = _velocity_matrix3(cell_velocity, n)
        for f in 1:nf
            i = Int(c1[f])
            j = Int(c2[f])
            if j > 0
                vx[f] = T(0.5 * (cm[i, 1] + cm[j, 1]))
                vy[f] = T(0.5 * (cm[i, 2] + cm[j, 2]))
                vz[f] = T(0.5 * (cm[i, 3] + cm[j, 3]))
            else
                vx[f] = T(cm[i, 1])
                vy[f] = T(cm[i, 2])
                vz[f] = T(cm[i, 3])
            end
        end
    end
    return vx, vy, vz
end

"""
    cartesian_periodic_mesh_arrays_3d(n; T=Float64, index_type=Int32,
                                      face_velocity=nothing,
                                      cell_velocity=nothing)

Build the face-table geometry for an `n^3` periodic Cartesian mesh in the unit
box.  This is the regular-lattice Voronoi limit and is useful as the first 3-D
GPU parity gate before feeding AREPO's fully unstructured 3-D face export.
"""
function cartesian_periodic_mesh_arrays_3d(n::Integer; T::Type{<:AbstractFloat} = Float64,
                                           index_type::Type{<:Integer} = Int32,
                                           face_velocity = nothing,
                                           cell_velocity = nothing)
    n > 0 || error("n must be positive")
    nc = n^3
    nf = 3nc
    c1 = Vector{index_type}(undef, nf)
    c2 = Vector{index_type}(undef, nf)
    nx = zeros(T, nf)
    ny = zeros(T, nf)
    nz = zeros(T, nf)
    area = fill(T((1 / n)^2), nf)
    f = 1
    @inbounds for k in 1:n, j in 1:n, i in 1:n
        id = _cell_id_3d(i, j, k, n)
        c1[f] = index_type(id); c2[f] = index_type(_cell_id_3d(_wrap1(i + 1, n), j, k, n))
        nx[f] = one(T); f += 1
        c1[f] = index_type(id); c2[f] = index_type(_cell_id_3d(i, _wrap1(j + 1, n), k, n))
        ny[f] = one(T); f += 1
        c1[f] = index_type(id); c2[f] = index_type(_cell_id_3d(i, j, _wrap1(k + 1, n), n))
        nz[f] = one(T); f += 1
    end
    offsets, faces, signs = _cell_face_csr(nc, c1, c2, index_type)
    fvx, fvy, fvz = _face_velocity_arrays_3d(c1, c2, face_velocity, cell_velocity, T)
    return ArepoMeshArrays3D(c1, c2, offsets, faces, signs, fill(T(1 / nc), nc),
                             area, nx, ny, nz, fvx, fvy, fvz)
end

function _ring_area_normal(vertices::AbstractMatrix)
    sx = 0.0; sy = 0.0; sz = 0.0
    nv = size(vertices, 1)
    @inbounds for i in 1:nv
        j = i == nv ? 1 : i + 1
        ax, ay, az = vertices[i, 1], vertices[i, 2], vertices[i, 3]
        bx, by, bz = vertices[j, 1], vertices[j, 2], vertices[j, 3]
        sx += ay * bz - az * by
        sy += az * bx - ax * bz
        sz += ax * by - ay * bx
    end
    return (sx, sy, sz)
end

"""
    arepo_voronoi_mesh_arrays_3d(geometry, volume; ...)

Convert an AREPO-style 3-D Voronoi face export to `ArepoMeshArrays3D`.  The
`geometry` object must provide `c1`, `c2`, `nv`, `normals`, and `verts`, matching
the local `ArepoLib.get_voronoi_3d` convention.  Cell ids are expected to be
1-based; nonpositive `c2` values are treated as boundary/foreign faces.
"""
function arepo_voronoi_mesh_arrays_3d(geometry, volume;
                                      T::Type{<:AbstractFloat} = Float64,
                                      index_type::Type{<:Integer} = Int32,
                                      face_velocity = nothing,
                                      cell_velocity = nothing)
    c1h = Int.(geometry.c1)
    c2h = Int.(geometry.c2)
    nc = length(volume)
    normals = Matrix{Float64}(geometry.normals)
    @inbounds for f in eachindex(c1h)
        if !(1 <= c1h[f] <= nc) && (1 <= c2h[f] <= nc)
            c1h[f], c2h[f] = c2h[f], c1h[f]
            normals[f, 1] = -normals[f, 1]
            normals[f, 2] = -normals[f, 2]
            normals[f, 3] = -normals[f, 3]
        end
    end
    all(1 .<= c1h .<= nc) || error("each face must touch at least one 1-based local cell")
    any((c2h .> nc)) && error("geometry.c2 contains a cell id larger than volume length")
    offsets_ring = vcat(0, cumsum(Int.(geometry.nv)))
    nf = length(c1h)
    area = Vector{T}(undef, nf)
    nx = Vector{T}(undef, nf)
    ny = Vector{T}(undef, nf)
    nz = Vector{T}(undef, nf)
    @inbounds for f in 1:nf
        lo = offsets_ring[f] + 1
        hi = offsets_ring[f + 1]
        sx, sy, sz = _ring_area_normal(@view geometry.verts[lo:hi, :])
        a = 0.5 * sqrt(sx * sx + sy * sy + sz * sz)
        gx, gy, gz = normals[f, 1], normals[f, 2], normals[f, 3]
        gn = sqrt(gx * gx + gy * gy + gz * gz)
        gn > 0 || error("zero face normal in AREPO geometry")
        area[f] = T(a)
        nx[f] = T(gx / gn)
        ny[f] = T(gy / gn)
        nz[f] = T(gz / gn)
    end
    offsets, faces, signs = _cell_face_csr(nc, c1h, c2h, index_type)
    fvx, fvy, fvz = _face_velocity_arrays_3d(c1h, c2h, face_velocity, cell_velocity, T)
    return ArepoMeshArrays3D(index_type.(c1h), index_type.(c2h), offsets, faces, signs,
                             T.(volume), area, nx, ny, nz, fvx, fvy, fvz)
end

function to_backend(be, mesh::ArepoMeshArrays3D; T::Type{<:AbstractFloat} = Float32,
                    index_type::Type{<:Integer} = Int32)
    return ArepoMeshArrays3D(
        _backend_copy(be, mesh.c1, index_type),
        _backend_copy(be, mesh.c2, index_type),
        _backend_copy(be, mesh.cell_face_offsets, index_type),
        _backend_copy(be, mesh.cell_faces, index_type),
        _backend_copy(be, mesh.cell_face_signs, index_type),
        _backend_copy(be, mesh.volume, T),
        _backend_copy(be, mesh.face_area, T),
        _backend_copy(be, mesh.normal_x, T),
        _backend_copy(be, mesh.normal_y, T),
        _backend_copy(be, mesh.normal_z, T),
        _backend_copy(be, mesh.face_vx, T),
        _backend_copy(be, mesh.face_vy, T),
        _backend_copy(be, mesh.face_vz, T),
    )
end

function to_backend(be, state::EulerState3D; T::Type{<:AbstractFloat} = Float32)
    return EulerState3D(_backend_copy(be, state.D, T),
                        _backend_copy(be, state.Mx, T),
                        _backend_copy(be, state.My, T),
                        _backend_copy(be, state.Mz, T),
                        _backend_copy(be, state.E, T))
end

function euler_state_3d(mesh::ArepoMeshArrays3D; rho = 1.0, vx = 0.0, vy = 0.0,
                        vz = 0.0, pressure = 1.0, gamma::Real = 5/3,
                        T::Type{<:AbstractFloat} = Float64)
    n = length(mesh.volume)
    r = _expand_cell_value(rho, n, T)
    ux = _expand_cell_value(vx, n, T)
    uy = _expand_cell_value(vy, n, T)
    uz = _expand_cell_value(vz, n, T)
    p = _expand_cell_value(pressure, n, T)
    D = copy(r)
    Mx = r .* ux
    My = r .* uy
    Mz = r .* uz
    E = p ./ T(gamma - 1) .+ T(0.5) .* r .* (ux .* ux .+ uy .* uy .+ uz .* uz)
    return EulerState3D(D, Mx, My, Mz, E)
end

function primitive_to_conserved_3d!(state::EulerState3D, rho, vx, vy, vz, pressure;
                                    gamma::Real)
    n = length(state.D)
    length(rho) == n && length(vx) == n && length(vy) == n &&
        length(vz) == n && length(pressure) == n ||
        error("primitive arrays must all match state length")
    T = eltype(state.D)
    @inbounds for i in 1:n
        r = T(rho[i])
        ux = T(vx[i])
        uy = T(vy[i])
        uz = T(vz[i])
        p = T(pressure[i])
        state.D[i] = r
        state.Mx[i] = r * ux
        state.My[i] = r * uy
        state.Mz[i] = r * uz
        state.E[i] = p / T(gamma - 1) + T(0.5) * r * (ux * ux + uy * uy + uz * uz)
    end
    return state
end

function conserved_to_primitive_3d(state::EulerState3D; gamma::Real)
    D = Array(state.D)
    Mx = Array(state.Mx)
    My = Array(state.My)
    Mz = Array(state.Mz)
    E = Array(state.E)
    vx = Mx ./ D
    vy = My ./ D
    vz = Mz ./ D
    pressure = (gamma - 1) .* (E .- 0.5 .* (Mx .* Mx .+ My .* My .+ Mz .* Mz) ./ D)
    return (; rho = D, vx, vy, vz, pressure)
end

primitive_work_3d(state::EulerState3D) =
    PrimitiveState3D((similar(state.D) for _ in 1:5)...)

@kernel function _conserved_to_primitive_3d_k!(rho, vx, vy, vz, pressure,
                                               @Const(D), @Const(Mx),
                                               @Const(My), @Const(Mz),
                                               @Const(E), gamma)
    i = @index(Global, Linear)
    T = eltype(rho)
    @inbounds begin
        r = D[i]
        ux = Mx[i] / r
        uy = My[i] / r
        uz = Mz[i] / r
        rho[i] = r
        vx[i] = ux
        vy[i] = uy
        vz[i] = uz
        pressure[i] = (gamma - one(T)) *
                      (E[i] - T(0.5) * (Mx[i] * ux + My[i] * uy + Mz[i] * uz))
    end
end

function conserved_to_primitive_3d!(out::PrimitiveState3D, state::EulerState3D;
                                    gamma::Real, synchronize::Bool = true)
    n = length(state.D)
    length(out.rho) == n && length(out.vx) == n && length(out.vy) == n &&
        length(out.vz) == n && length(out.pressure) == n ||
        error("primitive work arrays must match state length")
    be = KA.get_backend(state.D)
    T = eltype(state.D)
    _conserved_to_primitive_3d_k!(be)(out.rho, out.vx, out.vy, out.vz,
                                      out.pressure, state.D, state.Mx,
                                      state.My, state.Mz, state.E, T(gamma);
                                      ndrange = n)
    synchronize && KA.synchronize(be)
    return out
end

primitive_to_arrays_3d(prim::PrimitiveState3D) =
    (; rho = Array(prim.rho), vx = Array(prim.vx), vy = Array(prim.vy),
     vz = Array(prim.vz), pressure = Array(prim.pressure))

hydro_work_3d(state::EulerState3D, mesh::ArepoMeshArrays3D) =
    FaceFluxWork3D(similar(state.D, length(mesh.c1)),
                   similar(state.D, length(mesh.c1)),
                   similar(state.D, length(mesh.c1)),
                   similar(state.D, length(mesh.c1)),
                   similar(state.D, length(mesh.c1)))

face_prediction_work_3d(mesh::ArepoMeshArrays3D) =
    FaceStates3D(similar(mesh.face_area, 5 * length(mesh.face_area)),
                 similar(mesh.face_area, 5 * length(mesh.face_area)))

function face_states_to_arrays(states::FaceStates3D)
    nf = length(states.left) ÷ 5
    left = reshape(Array(states.left), nf, 5)
    right = reshape(Array(states.right), nf, 5)
    return (; left = (; rho = left[:, 1], vx = left[:, 2], vy = left[:, 3],
                      vz = left[:, 4], pressure = left[:, 5]),
            right = (; rho = right[:, 1], vx = right[:, 2], vy = right[:, 3],
                       vz = right[:, 4], pressure = right[:, 5]))
end

@inline function _pressure_3d(D::T, Mx::T, My::T, Mz::T, E::T, gm1::T,
                              small::T) where {T}
    kinetic = T(0.5) * (Mx * Mx + My * My + Mz * Mz) / D
    return max(gm1 * (E - kinetic), small)
end

@inline function _normal_flux_3d(D::T, Mx::T, My::T, Mz::T, E::T,
                                 nx::T, ny::T, nz::T, wx::T, wy::T, wz::T,
                                 gamma::T, gm1::T, small::T) where {T}
    vx = Mx / D
    vy = My / D
    vz = Mz / D
    p = _pressure_3d(D, Mx, My, Mz, E, gm1, small)
    un = vx * nx + vy * ny + vz * nz
    wn = wx * nx + wy * ny + wz * nz
    urel = un - wn
    return (D * urel,
            Mx * urel + p * nx,
            My * urel + p * ny,
            Mz * urel + p * nz,
            E * urel + p * un,
            urel,
            sqrt(gamma * p / D))
end

@inline function _hll_or_llf_flux_3d(Dl::T, Mxl::T, Myl::T, Mzl::T, El::T,
                                     Dr::T, Mxr::T, Myr::T, Mzr::T, Er::T,
                                     nx::T, ny::T, nz::T, wx::T, wy::T, wz::T,
                                     gamma::T, solver::Int, small::T) where {T}
    gm1 = gamma - one(T)
    FL = _normal_flux_3d(Dl, Mxl, Myl, Mzl, El, nx, ny, nz, wx, wy, wz,
                         gamma, gm1, small)
    FR = _normal_flux_3d(Dr, Mxr, Myr, Mzr, Er, nx, ny, nz, wx, wy, wz,
                         gamma, gm1, small)
    if solver == 1
        a = max(abs(FL[6]) + FL[7], abs(FR[6]) + FR[7])
        h = T(0.5)
        return (h * (FL[1] + FR[1] - a * (Dr  - Dl)),
                h * (FL[2] + FR[2] - a * (Mxr - Mxl)),
                h * (FL[3] + FR[3] - a * (Myr - Myl)),
                h * (FL[4] + FR[4] - a * (Mzr - Mzl)),
                h * (FL[5] + FR[5] - a * (Er  - El)))
    end
    sl = min(FL[6] - FL[7], FR[6] - FR[7])
    sr = max(FL[6] + FL[7], FR[6] + FR[7])
    sl >= zero(T) && return (FL[1], FL[2], FL[3], FL[4], FL[5])
    sr <= zero(T) && return (FR[1], FR[2], FR[3], FR[4], FR[5])
    denom = sr - sl
    return ((sr * FL[1] - sl * FR[1] + sl * sr * (Dr  - Dl)) / denom,
            (sr * FL[2] - sl * FR[2] + sl * sr * (Mxr - Mxl)) / denom,
            (sr * FL[3] - sl * FR[3] + sl * sr * (Myr - Myl)) / denom,
            (sr * FL[4] - sl * FR[4] + sl * sr * (Mzr - Mzl)) / denom,
            (sr * FL[5] - sl * FR[5] + sl * sr * (Er  - El)) / denom)
end

@kernel function _face_flux_3d_k!(FD, FMx, FMy, FMz, FE,
                                  @Const(D), @Const(Mx), @Const(My), @Const(Mz),
                                  @Const(E), @Const(c1), @Const(c2),
                                  @Const(area), @Const(nx), @Const(ny), @Const(nz),
                                  @Const(wx), @Const(wy), @Const(wz),
                                  gamma, solver::Int, small)
    f = @index(Global, Linear)
    T = eltype(FD)
    @inbounds begin
        i = Int(c1[f])
        j = Int(c2[f])
        if j <= 0
            FD[f] = zero(T); FMx[f] = zero(T); FMy[f] = zero(T)
            FMz[f] = zero(T); FE[f] = zero(T)
        else
            flux = _hll_or_llf_flux_3d(D[i], Mx[i], My[i], Mz[i], E[i],
                                       D[j], Mx[j], My[j], Mz[j], E[j],
                                       nx[f], ny[f], nz[f], wx[f], wy[f], wz[f],
                                       gamma, solver, small)
            a = area[f]
            FD[f] = flux[1] * a
            FMx[f] = flux[2] * a
            FMy[f] = flux[3] * a
            FMz[f] = flux[4] * a
            FE[f] = flux[5] * a
        end
    end
end

@kernel function _cell_update_3d_k!(D, Mx, My, Mz, E,
                                    @Const(FD), @Const(FMx), @Const(FMy),
                                    @Const(FMz), @Const(FE),
                                    @Const(old_volume), @Const(new_volume),
                                    @Const(offsets), @Const(faces),
                                    @Const(signs), dt)
    i = @index(Global, Linear)
    T = eltype(D)
    dD = zero(T); dMx = zero(T); dMy = zero(T); dMz = zero(T); dE = zero(T)
    @inbounds begin
        for p in Int(offsets[i]):(Int(offsets[i + 1]) - 1)
            f = Int(faces[p])
            s = T(signs[p])
            dD += s * FD[f]
            dMx += s * FMx[f]
            dMy += s * FMy[f]
            dMz += s * FMz[f]
            dE += s * FE[f]
        end
        vold = old_volume[i]
        vnew = new_volume[i]
        D[i] = (D[i] * vold + dt * dD) / vnew
        Mx[i] = (Mx[i] * vold + dt * dMx) / vnew
        My[i] = (My[i] * vold + dt * dMy) / vnew
        Mz[i] = (Mz[i] * vold + dt * dMz) / vnew
        E[i] = (E[i] * vold + dt * dE) / vnew
    end
end

@kernel function _cell_update_activecells_3d_k!(D, Mx, My, Mz, E,
                                                @Const(FD), @Const(FMx),
                                                @Const(FMy), @Const(FMz),
                                                @Const(FE),
                                                @Const(old_volume),
                                                @Const(new_volume),
                                                @Const(active_counts),
                                                @Const(active_faces),
                                                @Const(active_signs),
                                                active_stride, dt)
    i = @index(Global, Linear)
    T = eltype(D)
    dD = zero(T); dMx = zero(T); dMy = zero(T); dMz = zero(T); dE = zero(T)
    @inbounds begin
        base = (i - 1) * Int(active_stride)
        for q in 1:Int(active_counts[i])
            p = base + q
            f = Int(active_faces[p])
            s = T(active_signs[p])
            dD += s * FD[f]
            dMx += s * FMx[f]
            dMy += s * FMy[f]
            dMz += s * FMz[f]
            dE += s * FE[f]
        end
        vold = old_volume[i]
        vnew = new_volume[i]
        D[i] = (D[i] * vold + dt * dD) / vnew
        Mx[i] = (Mx[i] * vold + dt * dMx) / vnew
        My[i] = (My[i] * vold + dt * dMy) / vnew
        Mz[i] = (Mz[i] * vold + dt * dMz) / vnew
        E[i] = (E[i] * vold + dt * dE) / vnew
    end
end

function finite_volume_step_3d!(state::EulerState3D, mesh::ArepoMeshArrays3D;
                                dt::Real, gamma::Real, riemann::Symbol = :hll,
                                work::Union{Nothing,FaceFluxWork3D} = nothing,
                                new_volume = mesh.volume,
                                small_pressure::Real = 1e-12)
    be = KA.get_backend(state.D)
    T = eltype(state.D)
    w = work === nothing ? hydro_work_3d(state, mesh) : work
    _face_flux_3d_k!(be)(w.FD, w.FMx, w.FMy, w.FMz, w.FE,
                         state.D, state.Mx, state.My, state.Mz, state.E,
                         mesh.c1, mesh.c2, mesh.face_area,
                         mesh.normal_x, mesh.normal_y, mesh.normal_z,
                         mesh.face_vx, mesh.face_vy, mesh.face_vz,
                         T(gamma), _solver_code(riemann), T(small_pressure);
                         ndrange = length(mesh.c1))
    _cell_update_3d_k!(be)(state.D, state.Mx, state.My, state.Mz, state.E,
                           w.FD, w.FMx, w.FMy, w.FMz, w.FE,
                           mesh.volume, new_volume, mesh.cell_face_offsets,
                           mesh.cell_faces, mesh.cell_face_signs, T(dt);
                           ndrange = length(state.D))
    KA.synchronize(be)
    return state
end

@inline function _prim_to_cons3(rho::T, vx::T, vy::T, vz::T, p::T,
                                gamma::T) where {T}
    return (rho, rho * vx, rho * vy, rho * vz,
            p / (gamma - one(T)) + T(0.5) * rho * (vx * vx + vy * vy + vz * vz))
end

@kernel function _face_flux_from_predicted_3d_k!(FD, FMx, FMy, FMz, FE,
                                                 @Const(left), @Const(right),
                                                 @Const(c2), @Const(area),
                                                 @Const(nx), @Const(ny), @Const(nz),
                                                 @Const(wx), @Const(wy), @Const(wz),
                                                 gamma, solver::Int, small)
    f = @index(Global, Linear)
    T = eltype(FD)
    nface = Int(length(FD))
    @inbounds begin
        if Int(c2[f]) <= 0
            FD[f] = zero(T); FMx[f] = zero(T); FMy[f] = zero(T)
            FMz[f] = zero(T); FE[f] = zero(T)
        else
            # `predict_face_states_3d!` follows AREPO and stores velocities in
            # the face frame. Convert them back to the lab frame before using
            # the ALE flux, which subtracts the face velocity once.
            Dl = max(left[f], small)
            ulx = left[nface + f] + wx[f]
            uly = left[2 * nface + f] + wy[f]
            ulz = left[3 * nface + f] + wz[f]
            pl = max(left[4 * nface + f], small)
            Dr = max(right[f], small)
            urx = right[nface + f] + wx[f]
            ury = right[2 * nface + f] + wy[f]
            urz = right[3 * nface + f] + wz[f]
            pr = max(right[4 * nface + f], small)
            CL = _prim_to_cons3(Dl, ulx, uly, ulz, pl, gamma)
            CR = _prim_to_cons3(Dr, urx, ury, urz, pr, gamma)
            flux = _hll_or_llf_flux_3d(CL[1], CL[2], CL[3], CL[4], CL[5],
                                       CR[1], CR[2], CR[3], CR[4], CR[5],
                                       nx[f], ny[f], nz[f], wx[f], wy[f], wz[f],
                                       gamma, solver, small)
            a = area[f]
            FD[f] = flux[1] * a
            FMx[f] = flux[2] * a
            FMy[f] = flux[3] * a
            FMz[f] = flux[4] * a
            FE[f] = flux[5] * a
        end
    end
end

"""
    finite_volume_reconstructed_step_3d!(state, mesh, gradients, center, face_center; dt, gamma, ...)

Second-order AREPO-shaped hydro rung: predict primitive left/right face states
from limited gradients, evolve them by a half-step in the moving face frame,
then apply the same ALE HLL/LLF update as `finite_volume_step_3d!`.

This reproduces AREPO's reconstruction/predictor structure on an already-built
3-D face table. It intentionally does not rebuild the mesh.
"""
function finite_volume_reconstructed_step_3d!(
    state::EulerState3D, mesh::ArepoMeshArrays3D, gradients,
    center, face_center;
    dt::Real, gamma::Real, riemann::Symbol = :hll,
    dt_extrapolation = nothing,
    work::Union{Nothing,FaceFluxWork3D} = nothing,
    states::Union{Nothing,FaceStates3D} = nothing,
    new_volume = mesh.volume,
    box_size::Real = 1.0,
    small_pressure::Real = 1e-12,
    synchronize::Bool = true)
    prim = conserved_to_primitive_3d(state; gamma)
    n = length(prim.rho)
    half_dt = dt_extrapolation === nothing ? fill(eltype(prim.rho)(0.5 * dt), n) :
              dt_extrapolation
    s = states === nothing ? face_prediction_work_3d(mesh) : states
    predict_face_states_3d!(s, mesh, gradients, prim.rho, prim.vx, prim.vy, prim.vz,
                            prim.pressure, center, face_center;
                            dt_extrapolation = half_dt,
                            box_size, gamma)
    be = KA.get_backend(state.D)
    T = eltype(state.D)
    w = work === nothing ? hydro_work_3d(state, mesh) : work
    _face_flux_from_predicted_3d_k!(be)(w.FD, w.FMx, w.FMy, w.FMz, w.FE,
                                        s.left, s.right, mesh.c2, mesh.face_area,
                                        mesh.normal_x, mesh.normal_y, mesh.normal_z,
                                        mesh.face_vx, mesh.face_vy, mesh.face_vz,
                                        T(gamma), _solver_code(riemann),
                                        T(small_pressure);
                                        ndrange = length(mesh.c1))
    _cell_update_3d_k!(be)(state.D, state.Mx, state.My, state.Mz, state.E,
                           w.FD, w.FMx, w.FMy, w.FMz, w.FE,
                           mesh.volume, new_volume, mesh.cell_face_offsets,
                           mesh.cell_faces, mesh.cell_face_signs, T(dt);
                           ndrange = length(state.D))
    synchronize && KA.synchronize(be)
    return state
end

function finite_volume_reconstructed_step_3d!(
    state::EulerState3D, mesh::ArepoMeshArrays3D, gradients,
    prim::PrimitiveState3D, center, face_center;
    dt::Real, gamma::Real, riemann::Symbol = :hll,
    dt_extrapolation = nothing,
    work::Union{Nothing,FaceFluxWork3D} = nothing,
    states::Union{Nothing,FaceStates3D} = nothing,
    new_volume = mesh.volume,
    box_size::Real = 1.0,
    small_pressure::Real = 1e-12,
    synchronize::Bool = true)
    n = length(prim.rho)
    be = KA.get_backend(state.D)
    T = eltype(state.D)
    half_dt = dt_extrapolation === nothing ? fill(T(0.5 * dt), n) :
              dt_extrapolation
    s = states === nothing ? face_prediction_work_3d(mesh) : states
    predict_face_states_3d!(s, mesh, gradients, prim, center, face_center;
                            dt_extrapolation = half_dt,
                            box_size, gamma, synchronize = false)
    w = work === nothing ? hydro_work_3d(state, mesh) : work
    _face_flux_from_predicted_3d_k!(be)(w.FD, w.FMx, w.FMy, w.FMz, w.FE,
                                        s.left, s.right, mesh.c2, mesh.face_area,
                                        mesh.normal_x, mesh.normal_y, mesh.normal_z,
                                        mesh.face_vx, mesh.face_vy, mesh.face_vz,
                                        T(gamma), _solver_code(riemann),
                                        T(small_pressure);
                                        ndrange = length(mesh.c1))
    _cell_update_3d_k!(be)(state.D, state.Mx, state.My, state.Mz, state.E,
                           w.FD, w.FMx, w.FMy, w.FMz, w.FE,
                           mesh.volume, new_volume, mesh.cell_face_offsets,
                           mesh.cell_faces, mesh.cell_face_signs, T(dt);
                           ndrange = length(state.D))
    synchronize && KA.synchronize(be)
    return state
end

function finite_volume_reconstructed_step_3d!(
    state::EulerState3D, mesh::ArepoMeshArrays3D, gradients,
    prim::PrimitiveState3D, center_x, center_y, center_z,
    face_center_x, face_center_y, face_center_z;
    dt::Real, gamma::Real, riemann::Symbol = :hll,
    dt_extrapolation = nothing,
    work::Union{Nothing,FaceFluxWork3D} = nothing,
    states::Union{Nothing,FaceStates3D} = nothing,
    new_volume = mesh.volume,
    box_size::Real = 1.0,
    small_pressure::Real = 1e-12,
    synchronize::Bool = true)
    n = length(prim.rho)
    be = KA.get_backend(state.D)
    T = eltype(state.D)
    half_dt = dt_extrapolation === nothing ?
              _backend_copy(be, fill(T(0.5 * dt), n), T) :
              dt_extrapolation
    s = states === nothing ? face_prediction_work_3d(mesh) : states
    predict_face_states_3d!(s, mesh, gradients, prim,
                            center_x, center_y, center_z,
                            face_center_x, face_center_y, face_center_z;
                            dt_extrapolation = half_dt,
                            box_size, gamma, synchronize = false)
    w = work === nothing ? hydro_work_3d(state, mesh) : work
    _face_flux_from_predicted_3d_k!(be)(w.FD, w.FMx, w.FMy, w.FMz, w.FE,
                                        s.left, s.right, mesh.c2, mesh.face_area,
                                        mesh.normal_x, mesh.normal_y, mesh.normal_z,
                                        mesh.face_vx, mesh.face_vy, mesh.face_vz,
                                        T(gamma), _solver_code(riemann),
                                        T(small_pressure);
                                        ndrange = length(mesh.c1))
    _cell_update_3d_k!(be)(state.D, state.Mx, state.My, state.Mz, state.E,
                           w.FD, w.FMx, w.FMy, w.FMz, w.FE,
                           mesh.volume, new_volume, mesh.cell_face_offsets,
                           mesh.cell_faces, mesh.cell_face_signs, T(dt);
                           ndrange = length(state.D))
    synchronize && KA.synchronize(be)
    return state
end

function finite_volume_reconstructed_step_activecells_3d!(
    state::EulerState3D, mesh::ArepoMeshArrays3D, gradients,
    prim::PrimitiveState3D, center_x, center_y, center_z,
    face_center_x, face_center_y, face_center_z,
    active_counts, active_faces, active_signs;
    active_stride::Integer,
    dt::Real, gamma::Real, riemann::Symbol = :hll,
    dt_extrapolation = nothing,
    work::Union{Nothing,FaceFluxWork3D} = nothing,
    states::Union{Nothing,FaceStates3D} = nothing,
    new_volume = mesh.volume,
    box_size::Real = 1.0,
    small_pressure::Real = 1e-12,
    synchronize::Bool = true)
    n = length(prim.rho)
    be = KA.get_backend(state.D)
    T = eltype(state.D)
    half_dt = dt_extrapolation === nothing ?
              _backend_copy(be, fill(T(0.5 * dt), n), T) :
              dt_extrapolation
    s = states === nothing ? face_prediction_work_3d(mesh) : states
    predict_face_states_3d!(s, mesh, gradients, prim,
                            center_x, center_y, center_z,
                            face_center_x, face_center_y, face_center_z;
                            dt_extrapolation = half_dt,
                            box_size, gamma, synchronize = false)
    w = work === nothing ? hydro_work_3d(state, mesh) : work
    _face_flux_from_predicted_3d_k!(be)(w.FD, w.FMx, w.FMy, w.FMz, w.FE,
                                        s.left, s.right, mesh.c2, mesh.face_area,
                                        mesh.normal_x, mesh.normal_y, mesh.normal_z,
                                        mesh.face_vx, mesh.face_vy, mesh.face_vz,
                                        T(gamma), _solver_code(riemann),
                                        T(small_pressure);
                                        ndrange = length(mesh.c1))
    _cell_update_activecells_3d_k!(be)(
        state.D, state.Mx, state.My, state.Mz, state.E,
        w.FD, w.FMx, w.FMy, w.FMz, w.FE,
        mesh.volume, new_volume, active_counts, active_faces, active_signs,
        Int32(active_stride), T(dt);
        ndrange = length(state.D))
    synchronize && KA.synchronize(be)
    return state
end

"""
    arepo_hydro_dt_3d(volume, pressure, rho; gamma=5/3, courant=0.3, ...)

Continuous form of AREPO's non-cosmological hydro Courant rule:
`dt = CourantFac * cell_radius / sound_speed`, clipped by min/max timestep.
`cell_radius` is inferred from the 3-D cell volume as `(3V/4π)^(1/3)`.
"""
function arepo_hydro_dt_3d(volume, pressure, rho; gamma::Real = 5/3,
                           courant::Real = 0.3, max_dt::Real = 0.05,
                           min_dt::Real = 1e-6)
    cs = sqrt.(gamma .* pressure ./ rho)
    radius = cbrt.((3 .* volume) ./ (4pi))
    dt = courant .* radius ./ max.(cs, eps(eltype(float.(cs))))
    return clamp.(dt, min_dt, max_dt)
end

"""
    arepo_timebin_3d(dt; timebase_interval)

Map continuous timesteps to AREPO-style power-of-two integer bins. This mirrors
the non-cosmological hierarchy shape used after `get_timestep_hydro`; it is a
diagnostic helper for Julia gates, not a replacement for AREPO's full scheduler.
"""
function arepo_timebin_3d(dt; timebase_interval::Real)
    ti = max.(1, floor.(Int, dt ./ timebase_interval))
    bins = similar(ti)
    @inbounds for i in eachindex(ti)
        b = 0
        step = ti[i]
        while step > 1
            step >>= 1
            b += 1
        end
        bins[i] = b
    end
    return bins
end

function total_conserved_3d(state::EulerState3D, mesh::ArepoMeshArrays3D)
    D = Array(state.D); Mx = Array(state.Mx); My = Array(state.My)
    Mz = Array(state.Mz); E = Array(state.E); V = Array(mesh.volume)
    return (; mass = sum(D .* V),
            mx = sum(Mx .* V),
            my = sum(My .* V),
            mz = sum(Mz .* V),
            energy = sum(E .* V))
end

function max_signal_speed_3d(state::EulerState3D; gamma::Real)
    prim = conserved_to_primitive_3d(state; gamma)
    c = sqrt.(gamma .* max.(prim.pressure, 0) ./ prim.rho)
    return maximum(sqrt.(prim.vx .* prim.vx .+ prim.vy .* prim.vy .+
                         prim.vz .* prim.vz) .+ c)
end

struct GradientConnections3D{I<:AbstractVector,A<:AbstractVector}
    offsets::I
    cell_flags::I
    limit_offsets::I
    area::A
    center_other_x::A
    center_other_y::A
    center_other_z::A
    face_center_x::A
    face_center_y::A
    face_center_z::A
    limit_face_center_x::A
    limit_face_center_y::A
    limit_face_center_z::A
    rho_other::A
    velx_other::A
    vely_other::A
    velz_other::A
    press_other::A
end

struct HydroGradients3D{A<:AbstractVector}
    drho_x::A
    drho_y::A
    drho_z::A
    dvelx_x::A
    dvelx_y::A
    dvelx_z::A
    dvely_x::A
    dvely_y::A
    dvely_z::A
    dvelz_x::A
    dvelz_y::A
    dvelz_z::A
    dpress_x::A
    dpress_y::A
    dpress_z::A
end

function gradient_connections_3d(conn; T::Type{<:AbstractFloat} = Float64,
                                 index_type::Type{<:Integer} = Int32)
    return GradientConnections3D(
        index_type.(conn.offsets),
        index_type.(hasproperty(conn, :cell_flags) ? conn.cell_flags : zeros(Int, length(conn.offsets) - 1)),
        index_type.(hasproperty(conn, :limit_offsets) ? conn.limit_offsets : conn.offsets),
        T.(conn.area),
        T.(view(conn.center_other, :, 1)),
        T.(view(conn.center_other, :, 2)),
        T.(view(conn.center_other, :, 3)),
        T.(view(conn.face_center, :, 1)),
        T.(view(conn.face_center, :, 2)),
        T.(view(conn.face_center, :, 3)),
        T.(view(hasproperty(conn, :limit_face_center) ? conn.limit_face_center : conn.face_center, :, 1)),
        T.(view(hasproperty(conn, :limit_face_center) ? conn.limit_face_center : conn.face_center, :, 2)),
        T.(view(hasproperty(conn, :limit_face_center) ? conn.limit_face_center : conn.face_center, :, 3)),
        T.(conn.rho_other),
        T.(view(conn.vel_other, :, 1)),
        T.(view(conn.vel_other, :, 2)),
        T.(view(conn.vel_other, :, 3)),
        T.(conn.press_other),
    )
end

function gradient_connections_from_mesh_3d(mesh::ArepoMeshArrays3D, center,
                                           face_center, rho, velx, vely, velz,
                                           pressure;
                                           T::Type{<:AbstractFloat} = Float64,
                                           index_type::Type{<:Integer} = Int32)
    nc = length(mesh.volume)
    nf = length(mesh.c1)
    size(center) == (nc, 3) || error("center must be nc x 3")
    size(face_center) == (nf, 3) || error("face_center must be nf x 3")
    length(rho) == nc && length(velx) == nc && length(vely) == nc &&
        length(velz) == nc && length(pressure) == nc ||
        error("primitive arrays must match cell count")
    c1 = Int.(Array(mesh.c1))
    c2 = Int.(Array(mesh.c2))
    counts = zeros(Int, nc)
    @inbounds for f in 1:nf
        j = c2[f]
        j > 0 || continue
        counts[c1[f]] += 1
        counts[j] += 1
    end
    offsets = Vector{index_type}(undef, nc + 1)
    offsets[1] = one(index_type)
    @inbounds for i in 1:nc
        offsets[i + 1] = offsets[i] + index_type(counts[i])
    end
    cursor = Int.(offsets[1:end-1])
    nrows = Int(offsets[end] - one(index_type))
    area = Vector{T}(undef, nrows)
    center_other = Matrix{T}(undef, nrows, 3)
    fcenter = Matrix{T}(undef, nrows, 3)
    rho_other = Vector{T}(undef, nrows)
    vel_other = Matrix{T}(undef, nrows, 3)
    press_other = Vector{T}(undef, nrows)
    @inbounds for f in 1:nf
        i = c1[f]
        j = c2[f]
        j > 0 || continue
        for (self, other) in ((i, j), (j, i))
            row = cursor[self]
            cursor[self] += 1
            area[row] = T(mesh.face_area[f])
            center_other[row, 1] = T(center[other, 1])
            center_other[row, 2] = T(center[other, 2])
            center_other[row, 3] = T(center[other, 3])
            fcenter[row, 1] = T(face_center[f, 1])
            fcenter[row, 2] = T(face_center[f, 2])
            fcenter[row, 3] = T(face_center[f, 3])
            rho_other[row] = T(rho[other])
            vel_other[row, 1] = T(velx[other])
            vel_other[row, 2] = T(vely[other])
            vel_other[row, 3] = T(velz[other])
            press_other[row] = T(pressure[other])
        end
    end
    return gradient_connections_3d((; offsets,
        cell_flags = zeros(index_type, nc),
        limit_offsets = offsets,
        area,
        center_other,
        face_center = fcenter,
        limit_face_center = fcenter,
        rho_other,
        vel_other,
        press_other);
        T, index_type)
end

function to_backend(be, conn::GradientConnections3D; T::Type{<:AbstractFloat} = Float32,
                    index_type::Type{<:Integer} = Int32)
    return GradientConnections3D(
        _backend_copy(be, conn.offsets, index_type),
        _backend_copy(be, conn.cell_flags, index_type),
        _backend_copy(be, conn.limit_offsets, index_type),
        _backend_copy(be, conn.area, T),
        _backend_copy(be, conn.center_other_x, T),
        _backend_copy(be, conn.center_other_y, T),
        _backend_copy(be, conn.center_other_z, T),
        _backend_copy(be, conn.face_center_x, T),
        _backend_copy(be, conn.face_center_y, T),
        _backend_copy(be, conn.face_center_z, T),
        _backend_copy(be, conn.limit_face_center_x, T),
        _backend_copy(be, conn.limit_face_center_y, T),
        _backend_copy(be, conn.limit_face_center_z, T),
        _backend_copy(be, conn.rho_other, T),
        _backend_copy(be, conn.velx_other, T),
        _backend_copy(be, conn.vely_other, T),
        _backend_copy(be, conn.velz_other, T),
        _backend_copy(be, conn.press_other, T),
    )
end

hydro_gradient_work_3d(rho::AbstractVector) =
    HydroGradients3D((similar(rho) for _ in 1:15)...)

function _pack_gradient_columns(::Type{T}, cols...) where {T}
    n = length(first(cols))
    packed = Vector{T}(undef, n * length(cols))
    for (j, col) in enumerate(cols)
        length(col) == n || error("all packed columns must have the same length")
        offset = (j - 1) * n
        @inbounds for i in 1:n
            packed[offset + i] = T(col[i])
        end
    end
    return packed
end

@kernel function _pack_gradient_primitive_cell_data_3d_k!(
    cell_data, @Const(rho), @Const(velx), @Const(vely), @Const(velz),
    @Const(pressure), @Const(center_x), @Const(center_y), @Const(center_z),
    gamma)
    i = @index(Global, Linear)
    n = Int(length(rho))
    @inbounds begin
        cell_data[i] = rho[i]
        cell_data[n + i] = velx[i]
        cell_data[2 * n + i] = vely[i]
        cell_data[3 * n + i] = velz[i]
        cell_data[4 * n + i] = pressure[i]
        cell_data[5 * n + i] = sqrt(gamma * pressure[i] / rho[i])
        cell_data[6 * n + i] = center_x[i]
        cell_data[7 * n + i] = center_y[i]
        cell_data[8 * n + i] = center_z[i]
    end
end

@kernel function _pack_gradient_row_data_3d_k!(
    row_data, @Const(area), @Const(center_other_x), @Const(center_other_y),
    @Const(center_other_z), @Const(rho_other), @Const(velx_other),
    @Const(vely_other), @Const(velz_other), @Const(press_other))
    row = @index(Global, Linear)
    n = Int(length(area))
    @inbounds begin
        row_data[row] = area[row]
        row_data[n + row] = center_other_x[row]
        row_data[2 * n + row] = center_other_y[row]
        row_data[3 * n + row] = center_other_z[row]
        row_data[4 * n + row] = rho_other[row]
        row_data[5 * n + row] = velx_other[row]
        row_data[6 * n + row] = vely_other[row]
        row_data[7 * n + row] = velz_other[row]
        row_data[8 * n + row] = press_other[row]
    end
end

@kernel function _pack_gradient_limit_face_data_3d_k!(
    limit_face_data, @Const(limit_face_center_x),
    @Const(limit_face_center_y), @Const(limit_face_center_z))
    row = @index(Global, Linear)
    n = Int(length(limit_face_center_x))
    @inbounds begin
        limit_face_data[row] = limit_face_center_x[row]
        limit_face_data[n + row] = limit_face_center_y[row]
        limit_face_data[2 * n + row] = limit_face_center_z[row]
    end
end

@kernel function _pack_gradient_mesh_face_data_3d_k!(
    face_data, @Const(face_area), @Const(face_center_x),
    @Const(face_center_y), @Const(face_center_z))
    f = @index(Global, Linear)
    nf = Int(length(face_area))
    @inbounds begin
        face_data[f] = face_area[f]
        face_data[nf + f] = face_center_x[f]
        face_data[2 * nf + f] = face_center_y[f]
        face_data[3 * nf + f] = face_center_z[f]
    end
end

@kernel function _pack_predict_primitive_cell_data_3d_k!(
    cell_data, @Const(rho), @Const(velx), @Const(vely), @Const(velz),
    @Const(pressure), @Const(center_x), @Const(center_y), @Const(center_z),
    @Const(dt))
    i = @index(Global, Linear)
    n = Int(length(rho))
    @inbounds begin
        cell_data[i] = rho[i]
        cell_data[n + i] = velx[i]
        cell_data[2 * n + i] = vely[i]
        cell_data[3 * n + i] = velz[i]
        cell_data[4 * n + i] = pressure[i]
        cell_data[5 * n + i] = center_x[i]
        cell_data[6 * n + i] = center_y[i]
        cell_data[7 * n + i] = center_z[i]
        cell_data[8 * n + i] = dt[i]
    end
end

@kernel function _pack_predict_face_data_3d_k!(
    face_data, @Const(face_center_x), @Const(face_center_y),
    @Const(face_center_z), @Const(face_vx), @Const(face_vy), @Const(face_vz),
    @Const(normal_x), @Const(normal_y), @Const(normal_z))
    f = @index(Global, Linear)
    nf = Int(length(face_center_x))
    @inbounds begin
        face_data[f] = face_center_x[f]
        face_data[nf + f] = face_center_y[f]
        face_data[2 * nf + f] = face_center_z[f]
        face_data[3 * nf + f] = face_vx[f]
        face_data[4 * nf + f] = face_vy[f]
        face_data[5 * nf + f] = face_vz[f]
        face_data[6 * nf + f] = normal_x[f]
        face_data[7 * nf + f] = normal_y[f]
        face_data[8 * nf + f] = normal_z[f]
    end
end

function hydro_gradients_to_arrays(g::HydroGradients3D)
    drho = hcat(Array(g.drho_x), Array(g.drho_y), Array(g.drho_z))
    dpress = hcat(Array(g.dpress_x), Array(g.dpress_y), Array(g.dpress_z))
    n = length(g.drho_x)
    dvel = Array{eltype(drho)}(undef, n, 3, 3)
    dvel[:, 1, 1] .= Array(g.dvelx_x); dvel[:, 1, 2] .= Array(g.dvelx_y); dvel[:, 1, 3] .= Array(g.dvelx_z)
    dvel[:, 2, 1] .= Array(g.dvely_x); dvel[:, 2, 2] .= Array(g.dvely_y); dvel[:, 2, 3] .= Array(g.dvely_z)
    dvel[:, 3, 1] .= Array(g.dvelz_x); dvel[:, 3, 2] .= Array(g.dvelz_y); dvel[:, 3, 3] .= Array(g.dvelz_z)
    return (; drho, dvel, dpress)
end

@inline function _periodic_delta(d::T, box::T) where {T}
    box <= zero(T) && return d
    half = T(0.5) * box
    d < -half && return d + box
    d > half && return d - box
    return d
end

@inline function _solve_lsq3(x11::T, x12::T, x13::T, x22::T, x23::T, x33::T,
                             y1::T, y2::T, y3::T) where {T}
    x21 = x12; x31 = x13; x32 = x23
    p0 = 1; p1 = 2; p2 = 3
    if abs(x33) > abs(x22) && abs(x33) > abs(x11)
        p0 = 3; p1 = 1; p2 = 2
    elseif abs(x22) > abs(x11)
        p0 = 2; p1 = 1; p2 = 3
    end

    fac = -_xget3(x11, x12, x13, x21, x22, x23, x31, x32, x33, p1, p0) /
           _xget3(x11, x12, x13, x21, x22, x23, x31, x32, x33, p0, p0)
    x11, x12, x13, x21, x22, x23, x31, x32, x33, y1, y2, y3 =
        _add_row3(x11, x12, x13, x21, x22, x23, x31, x32, x33, y1, y2, y3, p0, fac, p1)

    fac = -_xget3(x11, x12, x13, x21, x22, x23, x31, x32, x33, p2, p0) /
           _xget3(x11, x12, x13, x21, x22, x23, x31, x32, x33, p0, p0)
    x11, x12, x13, x21, x22, x23, x31, x32, x33, y1, y2, y3 =
        _add_row3(x11, x12, x13, x21, x22, x23, x31, x32, x33, y1, y2, y3, p0, fac, p2)

    if abs(_xget3(x11, x12, x13, x21, x22, x23, x31, x32, x33, p1, p1)) <
       abs(_xget3(x11, x12, x13, x21, x22, x23, x31, x32, x33, p2, p2))
        tmp = p1; p1 = p2; p2 = tmp
    end

    fac = -_xget3(x11, x12, x13, x21, x22, x23, x31, x32, x33, p2, p1) /
           _xget3(x11, x12, x13, x21, x22, x23, x31, x32, x33, p1, p1)
    x11, x12, x13, x21, x22, x23, x31, x32, x33, y1, y2, y3 =
        _add_row3(x11, x12, x13, x21, x22, x23, x31, x32, x33, y1, y2, y3, p1, fac, p2)

    gp2 = _yget3(y1, y2, y3, p2) /
          _xget3(x11, x12, x13, x21, x22, x23, x31, x32, x33, p2, p2)
    gp1 = (_yget3(y1, y2, y3, p1) -
           _xget3(x11, x12, x13, x21, x22, x23, x31, x32, x33, p1, p2) * gp2) /
          _xget3(x11, x12, x13, x21, x22, x23, x31, x32, x33, p1, p1)
    gp0 = (_yget3(y1, y2, y3, p0) -
           _xget3(x11, x12, x13, x21, x22, x23, x31, x32, x33, p0, p1) * gp1 -
           _xget3(x11, x12, x13, x21, x22, x23, x31, x32, x33, p0, p2) * gp2) /
          _xget3(x11, x12, x13, x21, x22, x23, x31, x32, x33, p0, p0)

    g1 = zero(T); g2 = zero(T); g3 = zero(T)
    g1, g2, g3 = _gset3(g1, g2, g3, p2, gp2)
    g1, g2, g3 = _gset3(g1, g2, g3, p1, gp1)
    g1, g2, g3 = _gset3(g1, g2, g3, p0, gp0)
    return (g1, g2, g3)
end

@inline function _xget3(x11, x12, x13, x21, x22, x23, x31, x32, x33, r::Int, c::Int)
    r == 1 && return c == 1 ? x11 : c == 2 ? x12 : x13
    r == 2 && return c == 1 ? x21 : c == 2 ? x22 : x23
    return c == 1 ? x31 : c == 2 ? x32 : x33
end

@inline _yget3(y1, y2, y3, r::Int) = r == 1 ? y1 : r == 2 ? y2 : y3

@inline function _gset3(g1::T, g2::T, g3::T, r::Int, value::T) where {T}
    r == 1 && return (value, g2, g3)
    r == 2 && return (g1, value, g3)
    return (g1, g2, value)
end

@inline function _add_row3(x11::T, x12::T, x13::T, x21::T, x22::T, x23::T,
                           x31::T, x32::T, x33::T, y1::T, y2::T, y3::T,
                           source::Int, fac::T, target::Int) where {T}
    sy = _yget3(y1, y2, y3, source)
    sx1 = _xget3(x11, x12, x13, x21, x22, x23, x31, x32, x33, source, 1)
    sx2 = _xget3(x11, x12, x13, x21, x22, x23, x31, x32, x33, source, 2)
    sx3 = _xget3(x11, x12, x13, x21, x22, x23, x31, x32, x33, source, 3)
    if target == 1
        y1 += fac * sy
        x11 += fac * sx1; x12 += fac * sx2; x13 += fac * sx3
    elseif target == 2
        y2 += fac * sy
        x21 += fac * sx1; x22 += fac * sx2; x23 += fac * sx3
    else
        y3 += fac * sy
        x31 += fac * sx1; x32 += fac * sx2; x33 += fac * sx3
    end
    return (x11, x12, x13, x21, x22, x23, x31, x32, x33, y1, y2, y3)
end

@inline function _limit_gradient3(dx::T, dy::T, dz::T, phi::T, minphi::T,
                                  maxphi::T, gx::T, gy::T, gz::T) where {T}
    dp = gx * dx + gy * dy + gz * dz
    if dp > zero(T)
        if phi + dp > maxphi
            fac = maxphi > phi ? (maxphi - phi) / dp : zero(T)
            gx *= fac; gy *= fac; gz *= fac
        end
    elseif dp < zero(T)
        if phi + dp < minphi
            fac = minphi < phi ? (minphi - phi) / dp : zero(T)
            gx *= fac; gy *= fac; gz *= fac
        end
    end
    return (gx, gy, gz)
end

@inline function _limit_vel_gradient3(dx::T, dy::T, dz::T, csnd::T,
                                      gx::T, gy::T, gz::T) where {T}
    dv = abs(gx * dx + gy * dy + gz * dz)
    if dv > csnd
        fac = csnd / dv
        gx *= fac; gy *= fac; gz *= fac
    end
    return (gx, gy, gz)
end

@kernel function _gradients_3d_k!(
    grad_data,
    @Const(offsets), @Const(cell_flags), @Const(limit_offsets),
    @Const(cell_data), @Const(row_data), @Const(limit_face_data),
    box_size)
    i = @index(Global, Linear)
    T = eltype(cell_data)
    ncell = Int(length(offsets) - 1)
    nrow = Int(length(row_data) ÷ 9)
    nlimit = Int(length(limit_face_data) ÷ 3)
    x11 = zero(T); x12 = zero(T); x13 = zero(T)
    x22 = zero(T); x23 = zero(T); x33 = zero(T)
    yr1 = zero(T); yr2 = zero(T); yr3 = zero(T)
    yvx1 = zero(T); yvx2 = zero(T); yvx3 = zero(T)
    yvy1 = zero(T); yvy2 = zero(T); yvy3 = zero(T)
    yvz1 = zero(T); yvz2 = zero(T); yvz3 = zero(T)
    yp1 = zero(T); yp2 = zero(T); yp3 = zero(T)
    minr = typemax(T); maxr = -typemax(T)
    minvx = typemax(T); maxvx = -typemax(T)
    minvy = typemax(T); maxvy = -typemax(T)
    minvz = typemax(T); maxvz = -typemax(T)
    minp = typemax(T); maxp = -typemax(T)
    r0 = cell_data[i]
    vx0 = cell_data[ncell + i]
    vy0 = cell_data[2 * ncell + i]
    vz0 = cell_data[3 * ncell + i]
    p0 = cell_data[4 * ncell + i]
    cx = cell_data[6 * ncell + i]
    cy = cell_data[7 * ncell + i]
    cz = cell_data[8 * ncell + i]
    @inbounds begin
        for row in Int(offsets[i]):(Int(offsets[i + 1]) - 1)
            dx = _periodic_delta(row_data[nrow + row] - cx, box_size)
            dy = _periodic_delta(row_data[2 * nrow + row] - cy, box_size)
            dz = _periodic_delta(row_data[3 * nrow + row] - cz, box_size)
            dist = sqrt(dx * dx + dy * dy + dz * dz)
            invd = one(T) / dist
            nx = dx * invd; ny = dy * invd; nz = dz * invd
            w = row_data[row]
            x11 += w * nx * nx
            x12 += w * nx * ny
            x13 += w * nx * nz
            x22 += w * ny * ny
            x23 += w * ny * nz
            x33 += w * nz * nz

            ro = row_data[4 * nrow + row]
            vxo = row_data[5 * nrow + row]
            vyo = row_data[6 * nrow + row]
            vzo = row_data[7 * nrow + row]
            po = row_data[8 * nrow + row]

            fac = w * (ro - r0) * invd
            yr1 += fac * nx; yr2 += fac * ny; yr3 += fac * nz
            fac = w * (vxo - vx0) * invd
            yvx1 += fac * nx; yvx2 += fac * ny; yvx3 += fac * nz
            fac = w * (vyo - vy0) * invd
            yvy1 += fac * nx; yvy2 += fac * ny; yvy3 += fac * nz
            fac = w * (vzo - vz0) * invd
            yvz1 += fac * nx; yvz2 += fac * ny; yvz3 += fac * nz
            fac = w * (po - p0) * invd
            yp1 += fac * nx; yp2 += fac * ny; yp3 += fac * nz

            minr = min(minr, ro); maxr = max(maxr, ro)
            minvx = min(minvx, vxo); maxvx = max(maxvx, vxo)
            minvy = min(minvy, vyo); maxvy = max(maxvy, vyo)
            minvz = min(minvz, vzo); maxvz = max(maxvz, vzo)
            minp = min(minp, po); maxp = max(maxp, po)
        end

        gr = _solve_lsq3(x11, x12, x13, x22, x23, x33, yr1, yr2, yr3)
        gvx = _solve_lsq3(x11, x12, x13, x22, x23, x33, yvx1, yvx2, yvx3)
        gvy = _solve_lsq3(x11, x12, x13, x22, x23, x33, yvy1, yvy2, yvy3)
        gvz = _solve_lsq3(x11, x12, x13, x22, x23, x33, yvz1, yvz2, yvz3)
        gp = _solve_lsq3(x11, x12, x13, x22, x23, x33, yp1, yp2, yp3)

        grx = gr[1]; gry = gr[2]; grz = gr[3]
        gvxx = gvx[1]; gvxy = gvx[2]; gvxz = gvx[3]
        gvyx = gvy[1]; gvyy = gvy[2]; gvyz = gvy[3]
        gvzx = gvz[1]; gvzy = gvz[2]; gvzz = gvz[3]
        gpx = gp[1]; gpy = gp[2]; gpz = gp[3]

        flags = Int(cell_flags[i])
        if (flags & 1) != 0
            grx = zero(T); gvxx = zero(T); gvyx = zero(T); gvzx = zero(T); gpx = zero(T)
        end
        if (flags & 2) != 0
            gry = zero(T); gvxy = zero(T); gvyy = zero(T); gvzy = zero(T); gpy = zero(T)
        end
        if (flags & 4) != 0
            grz = zero(T); gvxz = zero(T); gvyz = zero(T); gvzz = zero(T); gpz = zero(T)
        end

        for row in Int(limit_offsets[i]):(Int(limit_offsets[i + 1]) - 1)
            dx = _periodic_delta(limit_face_data[row] - cx, box_size)
            dy = _periodic_delta(limit_face_data[nlimit + row] - cy, box_size)
            dz = _periodic_delta(limit_face_data[2 * nlimit + row] - cz, box_size)
            gr = _limit_gradient3(dx, dy, dz, r0, minr, maxr, grx, gry, grz)
            grx = gr[1]; gry = gr[2]; grz = gr[3]
            gvx = _limit_gradient3(dx, dy, dz, vx0, minvx, maxvx, gvxx, gvxy, gvxz)
            gvxx = gvx[1]; gvxy = gvx[2]; gvxz = gvx[3]
            gvy = _limit_gradient3(dx, dy, dz, vy0, minvy, maxvy, gvyx, gvyy, gvyz)
            gvyx = gvy[1]; gvyy = gvy[2]; gvyz = gvy[3]
            gvz = _limit_gradient3(dx, dy, dz, vz0, minvz, maxvz, gvzx, gvzy, gvzz)
            gvzx = gvz[1]; gvzy = gvz[2]; gvzz = gvz[3]
            gp = _limit_gradient3(dx, dy, dz, p0, minp, maxp, gpx, gpy, gpz)
            gpx = gp[1]; gpy = gp[2]; gpz = gp[3]
        end

        csnd = cell_data[5 * ncell + i]
        for row in Int(limit_offsets[i]):(Int(limit_offsets[i + 1]) - 1)
            dx = _periodic_delta(limit_face_data[row] - cx, box_size)
            dy = _periodic_delta(limit_face_data[nlimit + row] - cy, box_size)
            dz = _periodic_delta(limit_face_data[2 * nlimit + row] - cz, box_size)
            gvx = _limit_vel_gradient3(dx, dy, dz, csnd, gvxx, gvxy, gvxz)
            gvxx = gvx[1]; gvxy = gvx[2]; gvxz = gvx[3]
            gvy = _limit_vel_gradient3(dx, dy, dz, csnd, gvyx, gvyy, gvyz)
            gvyx = gvy[1]; gvyy = gvy[2]; gvyz = gvy[3]
            gvz = _limit_vel_gradient3(dx, dy, dz, csnd, gvzx, gvzy, gvzz)
            gvzx = gvz[1]; gvzy = gvz[2]; gvzz = gvz[3]
        end

        if (flags & 1) != 0
            grx = zero(T); gvxx = zero(T); gvyx = zero(T); gvzx = zero(T); gpx = zero(T)
        end
        if (flags & 2) != 0
            gry = zero(T); gvxy = zero(T); gvyy = zero(T); gvzy = zero(T); gpy = zero(T)
        end
        if (flags & 4) != 0
            grz = zero(T); gvxz = zero(T); gvyz = zero(T); gvzz = zero(T); gpz = zero(T)
        end

        grad_data[i] = grx
        grad_data[ncell + i] = gry
        grad_data[2 * ncell + i] = grz
        grad_data[3 * ncell + i] = gvxx
        grad_data[4 * ncell + i] = gvxy
        grad_data[5 * ncell + i] = gvxz
        grad_data[6 * ncell + i] = gvyx
        grad_data[7 * ncell + i] = gvyy
        grad_data[8 * ncell + i] = gvyz
        grad_data[9 * ncell + i] = gvzx
        grad_data[10 * ncell + i] = gvzy
        grad_data[11 * ncell + i] = gvzz
        grad_data[12 * ncell + i] = gpx
        grad_data[13 * ncell + i] = gpy
        grad_data[14 * ncell + i] = gpz
    end
end

@kernel function _unpack_gradients_3d_k!(
    drho_x, drho_y, drho_z,
    dvelx_x, dvelx_y, dvelx_z,
    dvely_x, dvely_y, dvely_z,
    dvelz_x, dvelz_y, dvelz_z,
    dpress_x, dpress_y, dpress_z,
    @Const(grad_data))
    i = @index(Global, Linear)
    ncell = Int(length(drho_x))
    @inbounds begin
        drho_x[i] = grad_data[i]
        drho_y[i] = grad_data[ncell + i]
        drho_z[i] = grad_data[2 * ncell + i]
        dvelx_x[i] = grad_data[3 * ncell + i]
        dvelx_y[i] = grad_data[4 * ncell + i]
        dvelx_z[i] = grad_data[5 * ncell + i]
        dvely_x[i] = grad_data[6 * ncell + i]
        dvely_y[i] = grad_data[7 * ncell + i]
        dvely_z[i] = grad_data[8 * ncell + i]
        dvelz_x[i] = grad_data[9 * ncell + i]
        dvelz_y[i] = grad_data[10 * ncell + i]
        dvelz_z[i] = grad_data[11 * ncell + i]
        dpress_x[i] = grad_data[12 * ncell + i]
        dpress_y[i] = grad_data[13 * ncell + i]
        dpress_z[i] = grad_data[14 * ncell + i]
    end
end

@kernel function _pack_hydro_gradients_3d_k!(
    grad_data, @Const(drho_x), @Const(drho_y), @Const(drho_z),
    @Const(dvelx_x), @Const(dvelx_y), @Const(dvelx_z),
    @Const(dvely_x), @Const(dvely_y), @Const(dvely_z),
    @Const(dvelz_x), @Const(dvelz_y), @Const(dvelz_z),
    @Const(dpress_x), @Const(dpress_y), @Const(dpress_z))
    i = @index(Global, Linear)
    n = Int(length(drho_x))
    @inbounds begin
        grad_data[i] = drho_x[i]
        grad_data[n + i] = drho_y[i]
        grad_data[2 * n + i] = drho_z[i]
        grad_data[3 * n + i] = dvelx_x[i]
        grad_data[4 * n + i] = dvelx_y[i]
        grad_data[5 * n + i] = dvelx_z[i]
        grad_data[6 * n + i] = dvely_x[i]
        grad_data[7 * n + i] = dvely_y[i]
        grad_data[8 * n + i] = dvely_z[i]
        grad_data[9 * n + i] = dvelz_x[i]
        grad_data[10 * n + i] = dvelz_y[i]
        grad_data[11 * n + i] = dvelz_z[i]
        grad_data[12 * n + i] = dpress_x[i]
        grad_data[13 * n + i] = dpress_y[i]
        grad_data[14 * n + i] = dpress_z[i]
    end
end

function _pack_hydro_gradients_backend(be, ::Type{T},
                                       gradients::HydroGradients3D) where {T}
    n = length(gradients.drho_x)
    grad_data = _backend_zeros(be, T, 15 * n)
    _pack_hydro_gradients_3d_k!(be)(
        grad_data, gradients.drho_x, gradients.drho_y, gradients.drho_z,
        gradients.dvelx_x, gradients.dvelx_y, gradients.dvelx_z,
        gradients.dvely_x, gradients.dvely_y, gradients.dvely_z,
        gradients.dvelz_x, gradients.dvelz_y, gradients.dvelz_z,
        gradients.dpress_x, gradients.dpress_y, gradients.dpress_z;
        ndrange = n)
    return grad_data
end

@kernel function _gradients_from_mesh_3d_k!(
    grad_data, @Const(offsets), @Const(cell_faces), @Const(cell_signs),
    @Const(c1), @Const(c2), @Const(cell_data), @Const(face_data), box_size)
    i = @index(Global, Linear)
    T = eltype(cell_data)
    ncell = Int(length(offsets) - 1)
    nface = Int(length(c1))
    x11 = zero(T); x12 = zero(T); x13 = zero(T)
    x22 = zero(T); x23 = zero(T); x33 = zero(T)
    yr1 = zero(T); yr2 = zero(T); yr3 = zero(T)
    yvx1 = zero(T); yvx2 = zero(T); yvx3 = zero(T)
    yvy1 = zero(T); yvy2 = zero(T); yvy3 = zero(T)
    yvz1 = zero(T); yvz2 = zero(T); yvz3 = zero(T)
    yp1 = zero(T); yp2 = zero(T); yp3 = zero(T)
    minr = typemax(T); maxr = -typemax(T)
    minvx = typemax(T); maxvx = -typemax(T)
    minvy = typemax(T); maxvy = -typemax(T)
    minvz = typemax(T); maxvz = -typemax(T)
    minp = typemax(T); maxp = -typemax(T)
    r0 = cell_data[i]
    vx0 = cell_data[ncell + i]
    vy0 = cell_data[2 * ncell + i]
    vz0 = cell_data[3 * ncell + i]
    p0 = cell_data[4 * ncell + i]
    csnd = cell_data[5 * ncell + i]
    cx = cell_data[6 * ncell + i]
    cy = cell_data[7 * ncell + i]
    cz = cell_data[8 * ncell + i]
    @inbounds begin
        for p in Int(offsets[i]):(Int(offsets[i + 1]) - 1)
            f = Int(cell_faces[p])
            w = face_data[f]
            w > zero(T) || continue
            sign = Int(cell_signs[p])
            other = sign < 0 ? Int(c2[f]) : Int(c1[f])
            other > 0 || continue
            dx = _periodic_delta(cell_data[6 * ncell + other] - cx, box_size)
            dy = _periodic_delta(cell_data[7 * ncell + other] - cy, box_size)
            dz = _periodic_delta(cell_data[8 * ncell + other] - cz, box_size)
            dist = sqrt(dx * dx + dy * dy + dz * dz)
            invd = one(T) / dist
            nx = dx * invd; ny = dy * invd; nz = dz * invd
            x11 += w * nx * nx
            x12 += w * nx * ny
            x13 += w * nx * nz
            x22 += w * ny * ny
            x23 += w * ny * nz
            x33 += w * nz * nz

            ro = cell_data[other]
            vxo = cell_data[ncell + other]
            vyo = cell_data[2 * ncell + other]
            vzo = cell_data[3 * ncell + other]
            po = cell_data[4 * ncell + other]

            fac = w * (ro - r0) * invd
            yr1 += fac * nx; yr2 += fac * ny; yr3 += fac * nz
            fac = w * (vxo - vx0) * invd
            yvx1 += fac * nx; yvx2 += fac * ny; yvx3 += fac * nz
            fac = w * (vyo - vy0) * invd
            yvy1 += fac * nx; yvy2 += fac * ny; yvy3 += fac * nz
            fac = w * (vzo - vz0) * invd
            yvz1 += fac * nx; yvz2 += fac * ny; yvz3 += fac * nz
            fac = w * (po - p0) * invd
            yp1 += fac * nx; yp2 += fac * ny; yp3 += fac * nz

            minr = min(minr, ro); maxr = max(maxr, ro)
            minvx = min(minvx, vxo); maxvx = max(maxvx, vxo)
            minvy = min(minvy, vyo); maxvy = max(maxvy, vyo)
            minvz = min(minvz, vzo); maxvz = max(maxvz, vzo)
            minp = min(minp, po); maxp = max(maxp, po)
        end

        gr = _solve_lsq3(x11, x12, x13, x22, x23, x33, yr1, yr2, yr3)
        gvx = _solve_lsq3(x11, x12, x13, x22, x23, x33, yvx1, yvx2, yvx3)
        gvy = _solve_lsq3(x11, x12, x13, x22, x23, x33, yvy1, yvy2, yvy3)
        gvz = _solve_lsq3(x11, x12, x13, x22, x23, x33, yvz1, yvz2, yvz3)
        gp = _solve_lsq3(x11, x12, x13, x22, x23, x33, yp1, yp2, yp3)

        grx = gr[1]; gry = gr[2]; grz = gr[3]
        gvxx = gvx[1]; gvxy = gvx[2]; gvxz = gvx[3]
        gvyx = gvy[1]; gvyy = gvy[2]; gvyz = gvy[3]
        gvzx = gvz[1]; gvzy = gvz[2]; gvzz = gvz[3]
        gpx = gp[1]; gpy = gp[2]; gpz = gp[3]

        for p in Int(offsets[i]):(Int(offsets[i + 1]) - 1)
            f = Int(cell_faces[p])
            face_data[f] > zero(T) || continue
            dx = _periodic_delta(face_data[nface + f] - cx, box_size)
            dy = _periodic_delta(face_data[2 * nface + f] - cy, box_size)
            dz = _periodic_delta(face_data[3 * nface + f] - cz, box_size)
            gr = _limit_gradient3(dx, dy, dz, r0, minr, maxr, grx, gry, grz)
            grx = gr[1]; gry = gr[2]; grz = gr[3]
            gvx = _limit_gradient3(dx, dy, dz, vx0, minvx, maxvx, gvxx, gvxy, gvxz)
            gvxx = gvx[1]; gvxy = gvx[2]; gvxz = gvx[3]
            gvy = _limit_gradient3(dx, dy, dz, vy0, minvy, maxvy, gvyx, gvyy, gvyz)
            gvyx = gvy[1]; gvyy = gvy[2]; gvyz = gvy[3]
            gvz = _limit_gradient3(dx, dy, dz, vz0, minvz, maxvz, gvzx, gvzy, gvzz)
            gvzx = gvz[1]; gvzy = gvz[2]; gvzz = gvz[3]
            gp = _limit_gradient3(dx, dy, dz, p0, minp, maxp, gpx, gpy, gpz)
            gpx = gp[1]; gpy = gp[2]; gpz = gp[3]
        end

        for p in Int(offsets[i]):(Int(offsets[i + 1]) - 1)
            f = Int(cell_faces[p])
            face_data[f] > zero(T) || continue
            dx = _periodic_delta(face_data[nface + f] - cx, box_size)
            dy = _periodic_delta(face_data[2 * nface + f] - cy, box_size)
            dz = _periodic_delta(face_data[3 * nface + f] - cz, box_size)
            gvx = _limit_vel_gradient3(dx, dy, dz, csnd, gvxx, gvxy, gvxz)
            gvxx = gvx[1]; gvxy = gvx[2]; gvxz = gvx[3]
            gvy = _limit_vel_gradient3(dx, dy, dz, csnd, gvyx, gvyy, gvyz)
            gvyx = gvy[1]; gvyy = gvy[2]; gvyz = gvy[3]
            gvz = _limit_vel_gradient3(dx, dy, dz, csnd, gvzx, gvzy, gvzz)
            gvzx = gvz[1]; gvzy = gvz[2]; gvzz = gvz[3]
        end

        grad_data[i] = grx
        grad_data[ncell + i] = gry
        grad_data[2 * ncell + i] = grz
        grad_data[3 * ncell + i] = gvxx
        grad_data[4 * ncell + i] = gvxy
        grad_data[5 * ncell + i] = gvxz
        grad_data[6 * ncell + i] = gvyx
        grad_data[7 * ncell + i] = gvyy
        grad_data[8 * ncell + i] = gvyz
        grad_data[9 * ncell + i] = gvzx
        grad_data[10 * ncell + i] = gvzy
        grad_data[11 * ncell + i] = gvzz
        grad_data[12 * ncell + i] = gpx
        grad_data[13 * ncell + i] = gpy
        grad_data[14 * ncell + i] = gpz
    end
end

@kernel function _gradients_from_mesh_activecells_3d_k!(
    grad_data, @Const(active_counts), @Const(active_faces),
    @Const(active_signs), active_stride, @Const(c1), @Const(c2),
    @Const(cell_data), @Const(face_data), box_size)
    i = @index(Global, Linear)
    T = eltype(cell_data)
    ncell = Int(length(active_counts))
    nface = Int(length(c1))
    x11 = zero(T); x12 = zero(T); x13 = zero(T)
    x22 = zero(T); x23 = zero(T); x33 = zero(T)
    yr1 = zero(T); yr2 = zero(T); yr3 = zero(T)
    yvx1 = zero(T); yvx2 = zero(T); yvx3 = zero(T)
    yvy1 = zero(T); yvy2 = zero(T); yvy3 = zero(T)
    yvz1 = zero(T); yvz2 = zero(T); yvz3 = zero(T)
    yp1 = zero(T); yp2 = zero(T); yp3 = zero(T)
    minr = typemax(T); maxr = -typemax(T)
    minvx = typemax(T); maxvx = -typemax(T)
    minvy = typemax(T); maxvy = -typemax(T)
    minvz = typemax(T); maxvz = -typemax(T)
    minp = typemax(T); maxp = -typemax(T)
    r0 = cell_data[i]
    vx0 = cell_data[ncell + i]
    vy0 = cell_data[2 * ncell + i]
    vz0 = cell_data[3 * ncell + i]
    p0 = cell_data[4 * ncell + i]
    csnd = cell_data[5 * ncell + i]
    cx = cell_data[6 * ncell + i]
    cy = cell_data[7 * ncell + i]
    cz = cell_data[8 * ncell + i]
    @inbounds begin
        base = (i - 1) * Int(active_stride)
        count = Int(active_counts[i])
        for q in 1:count
            p = base + q
            f = Int(active_faces[p])
            w = face_data[f]
            w > zero(T) || continue
            sign = Int(active_signs[p])
            other = sign < 0 ? Int(c2[f]) : Int(c1[f])
            other > 0 || continue
            dx = _periodic_delta(cell_data[6 * ncell + other] - cx, box_size)
            dy = _periodic_delta(cell_data[7 * ncell + other] - cy, box_size)
            dz = _periodic_delta(cell_data[8 * ncell + other] - cz, box_size)
            dist = sqrt(dx * dx + dy * dy + dz * dz)
            invd = one(T) / dist
            nx = dx * invd; ny = dy * invd; nz = dz * invd
            x11 += w * nx * nx
            x12 += w * nx * ny
            x13 += w * nx * nz
            x22 += w * ny * ny
            x23 += w * ny * nz
            x33 += w * nz * nz

            ro = cell_data[other]
            vxo = cell_data[ncell + other]
            vyo = cell_data[2 * ncell + other]
            vzo = cell_data[3 * ncell + other]
            po = cell_data[4 * ncell + other]

            fac = w * (ro - r0) * invd
            yr1 += fac * nx; yr2 += fac * ny; yr3 += fac * nz
            fac = w * (vxo - vx0) * invd
            yvx1 += fac * nx; yvx2 += fac * ny; yvx3 += fac * nz
            fac = w * (vyo - vy0) * invd
            yvy1 += fac * nx; yvy2 += fac * ny; yvy3 += fac * nz
            fac = w * (vzo - vz0) * invd
            yvz1 += fac * nx; yvz2 += fac * ny; yvz3 += fac * nz
            fac = w * (po - p0) * invd
            yp1 += fac * nx; yp2 += fac * ny; yp3 += fac * nz

            minr = min(minr, ro); maxr = max(maxr, ro)
            minvx = min(minvx, vxo); maxvx = max(maxvx, vxo)
            minvy = min(minvy, vyo); maxvy = max(maxvy, vyo)
            minvz = min(minvz, vzo); maxvz = max(maxvz, vzo)
            minp = min(minp, po); maxp = max(maxp, po)
        end

        gr = _solve_lsq3(x11, x12, x13, x22, x23, x33, yr1, yr2, yr3)
        gvx = _solve_lsq3(x11, x12, x13, x22, x23, x33, yvx1, yvx2, yvx3)
        gvy = _solve_lsq3(x11, x12, x13, x22, x23, x33, yvy1, yvy2, yvy3)
        gvz = _solve_lsq3(x11, x12, x13, x22, x23, x33, yvz1, yvz2, yvz3)
        gp = _solve_lsq3(x11, x12, x13, x22, x23, x33, yp1, yp2, yp3)

        grx = gr[1]; gry = gr[2]; grz = gr[3]
        gvxx = gvx[1]; gvxy = gvx[2]; gvxz = gvx[3]
        gvyx = gvy[1]; gvyy = gvy[2]; gvyz = gvy[3]
        gvzx = gvz[1]; gvzy = gvz[2]; gvzz = gvz[3]
        gpx = gp[1]; gpy = gp[2]; gpz = gp[3]

        for q in 1:count
            f = Int(active_faces[base + q])
            face_data[f] > zero(T) || continue
            dx = _periodic_delta(face_data[nface + f] - cx, box_size)
            dy = _periodic_delta(face_data[2 * nface + f] - cy, box_size)
            dz = _periodic_delta(face_data[3 * nface + f] - cz, box_size)
            gr = _limit_gradient3(dx, dy, dz, r0, minr, maxr, grx, gry, grz)
            grx = gr[1]; gry = gr[2]; grz = gr[3]
            gvx = _limit_gradient3(dx, dy, dz, vx0, minvx, maxvx, gvxx, gvxy, gvxz)
            gvxx = gvx[1]; gvxy = gvx[2]; gvxz = gvx[3]
            gvy = _limit_gradient3(dx, dy, dz, vy0, minvy, maxvy, gvyx, gvyy, gvyz)
            gvyx = gvy[1]; gvyy = gvy[2]; gvyz = gvy[3]
            gvz = _limit_gradient3(dx, dy, dz, vz0, minvz, maxvz, gvzx, gvzy, gvzz)
            gvzx = gvz[1]; gvzy = gvz[2]; gvzz = gvz[3]
            gp = _limit_gradient3(dx, dy, dz, p0, minp, maxp, gpx, gpy, gpz)
            gpx = gp[1]; gpy = gp[2]; gpz = gp[3]
        end

        for q in 1:count
            f = Int(active_faces[base + q])
            face_data[f] > zero(T) || continue
            dx = _periodic_delta(face_data[nface + f] - cx, box_size)
            dy = _periodic_delta(face_data[2 * nface + f] - cy, box_size)
            dz = _periodic_delta(face_data[3 * nface + f] - cz, box_size)
            gvx = _limit_vel_gradient3(dx, dy, dz, csnd, gvxx, gvxy, gvxz)
            gvxx = gvx[1]; gvxy = gvx[2]; gvxz = gvx[3]
            gvy = _limit_vel_gradient3(dx, dy, dz, csnd, gvyx, gvyy, gvyz)
            gvyx = gvy[1]; gvyy = gvy[2]; gvyz = gvy[3]
            gvz = _limit_vel_gradient3(dx, dy, dz, csnd, gvzx, gvzy, gvzz)
            gvzx = gvz[1]; gvzy = gvz[2]; gvzz = gvz[3]
        end

        grad_data[i] = grx
        grad_data[ncell + i] = gry
        grad_data[2 * ncell + i] = grz
        grad_data[3 * ncell + i] = gvxx
        grad_data[4 * ncell + i] = gvxy
        grad_data[5 * ncell + i] = gvxz
        grad_data[6 * ncell + i] = gvyx
        grad_data[7 * ncell + i] = gvyy
        grad_data[8 * ncell + i] = gvyz
        grad_data[9 * ncell + i] = gvzx
        grad_data[10 * ncell + i] = gvzy
        grad_data[11 * ncell + i] = gvzz
        grad_data[12 * ncell + i] = gpx
        grad_data[13 * ncell + i] = gpy
        grad_data[14 * ncell + i] = gpz
    end
end

function calculate_gradients_3d!(out::HydroGradients3D, conn::GradientConnections3D,
                                 rho, velx, vely, velz, pressure, center;
                                 box_size::Real = 1.0, gamma::Real = 5/3,
                                 sound_speed = nothing)
    n = length(rho)
    size(center) == (n, 3) || error("center must be n x 3")
    be = KA.get_backend(out.drho_x)
    T = eltype(out.drho_x)
    I = eltype(conn.offsets)
    offsets = _backend_copy(be, Array(conn.offsets), I)
    cell_flags = _backend_copy(be, Array(conn.cell_flags), I)
    limit_offsets = _backend_copy(be, Array(conn.limit_offsets), I)
    csnd = sound_speed === nothing ? sqrt.(T(gamma) .* Array(pressure) ./ Array(rho)) : sound_speed
    cell_data = _backend_copy(be,
        _pack_gradient_columns(T, rho, velx, vely, velz, pressure, csnd,
                               view(center, :, 1), view(center, :, 2), view(center, :, 3)), T)
    row_data = _backend_copy(be,
        _pack_gradient_columns(T, Array(conn.area), Array(conn.center_other_x),
                               Array(conn.center_other_y), Array(conn.center_other_z),
                               Array(conn.rho_other), Array(conn.velx_other),
                               Array(conn.vely_other), Array(conn.velz_other),
                               Array(conn.press_other)), T)
    limit_face_data = _backend_copy(be,
        _pack_gradient_columns(T, Array(conn.limit_face_center_x),
                               Array(conn.limit_face_center_y),
                               Array(conn.limit_face_center_z)), T)
    grad_data = _backend_zeros(be, T, 15 * n)
    _gradients_3d_k!(be)(grad_data, offsets, cell_flags, limit_offsets,
                         cell_data, row_data, limit_face_data, T(box_size);
                         ndrange = n)
    _unpack_gradients_3d_k!(be)(
        out.drho_x, out.drho_y, out.drho_z,
        out.dvelx_x, out.dvelx_y, out.dvelx_z,
        out.dvely_x, out.dvely_y, out.dvely_z,
        out.dvelz_x, out.dvelz_y, out.dvelz_z,
        out.dpress_x, out.dpress_y, out.dpress_z,
        grad_data;
        ndrange = n)
    KA.synchronize(be)
    return out
end

function calculate_gradients_3d!(out::HydroGradients3D, conn::GradientConnections3D,
                                 prim::PrimitiveState3D, center;
                                 box_size::Real = 1.0, gamma::Real = 5/3,
                                 sound_speed = nothing,
                                 synchronize::Bool = true)
    n = length(prim.rho)
    size(center) == (n, 3) || error("center must be n x 3")
    be = KA.get_backend(out.drho_x)
    T = eltype(out.drho_x)
    offsets = conn.offsets
    cell_flags = conn.cell_flags
    limit_offsets = conn.limit_offsets
    if sound_speed === nothing
        cx = _backend_copy(be, collect(view(center, :, 1)), T)
        cy = _backend_copy(be, collect(view(center, :, 2)), T)
        cz = _backend_copy(be, collect(view(center, :, 3)), T)
        cell_data = _backend_zeros(be, T, 9 * n)
        _pack_gradient_primitive_cell_data_3d_k!(be)(
            cell_data, prim.rho, prim.vx, prim.vy, prim.vz, prim.pressure,
            cx, cy, cz, T(gamma);
            ndrange = n)
    else
        cell_data = _backend_copy(be,
            _pack_gradient_columns(T, Array(prim.rho), Array(prim.vx),
                                   Array(prim.vy), Array(prim.vz),
                                   Array(prim.pressure), sound_speed,
                                   view(center, :, 1), view(center, :, 2),
                                   view(center, :, 3)), T)
    end
    nrow = length(conn.area)
    row_data = _backend_zeros(be, T, 9 * nrow)
    _pack_gradient_row_data_3d_k!(be)(
        row_data, conn.area, conn.center_other_x, conn.center_other_y,
        conn.center_other_z, conn.rho_other, conn.velx_other, conn.vely_other,
        conn.velz_other, conn.press_other;
        ndrange = nrow)
    nlimit = length(conn.limit_face_center_x)
    limit_face_data = _backend_zeros(be, T, 3 * nlimit)
    _pack_gradient_limit_face_data_3d_k!(be)(
        limit_face_data, conn.limit_face_center_x, conn.limit_face_center_y,
        conn.limit_face_center_z;
        ndrange = nlimit)
    grad_data = _backend_zeros(be, T, 15 * n)
    _gradients_3d_k!(be)(grad_data, offsets, cell_flags, limit_offsets,
                         cell_data, row_data, limit_face_data, T(box_size);
                         ndrange = n)
    _unpack_gradients_3d_k!(be)(
        out.drho_x, out.drho_y, out.drho_z,
        out.dvelx_x, out.dvelx_y, out.dvelx_z,
        out.dvely_x, out.dvely_y, out.dvely_z,
        out.dvelz_x, out.dvelz_y, out.dvelz_z,
        out.dpress_x, out.dpress_y, out.dpress_z,
        grad_data;
        ndrange = n)
    synchronize && KA.synchronize(be)
    return out
end

function calculate_gradients_from_mesh_3d!(out::HydroGradients3D,
                                           mesh::ArepoMeshArrays3D,
                                           prim::PrimitiveState3D,
                                           center, face_center;
                                           box_size::Real = 1.0,
                                           gamma::Real = 5/3,
                                           synchronize::Bool = true)
    n = length(prim.rho)
    nf = length(mesh.c1)
    size(center) == (n, 3) || error("center must be n x 3")
    size(face_center) == (nf, 3) || error("face_center must be nf x 3")
    be = KA.get_backend(out.drho_x)
    T = eltype(out.drho_x)
    cx = _backend_copy(be, collect(view(center, :, 1)), T)
    cy = _backend_copy(be, collect(view(center, :, 2)), T)
    cz = _backend_copy(be, collect(view(center, :, 3)), T)
    cell_data = _backend_zeros(be, T, 9 * n)
    _pack_gradient_primitive_cell_data_3d_k!(be)(
        cell_data, prim.rho, prim.vx, prim.vy, prim.vz, prim.pressure,
        cx, cy, cz, T(gamma);
        ndrange = n)
    fcx = _backend_copy(be, collect(view(face_center, :, 1)), T)
    fcy = _backend_copy(be, collect(view(face_center, :, 2)), T)
    fcz = _backend_copy(be, collect(view(face_center, :, 3)), T)
    face_data = _backend_zeros(be, T, 4 * nf)
    _pack_gradient_mesh_face_data_3d_k!(be)(
        face_data, mesh.face_area, fcx, fcy, fcz;
        ndrange = nf)
    grad_data = _backend_zeros(be, T, 15 * n)
    _gradients_from_mesh_3d_k!(be)(
        grad_data, mesh.cell_face_offsets, mesh.cell_faces,
        mesh.cell_face_signs, mesh.c1, mesh.c2, cell_data, face_data,
        T(box_size);
        ndrange = n)
    _unpack_gradients_3d_k!(be)(
        out.drho_x, out.drho_y, out.drho_z,
        out.dvelx_x, out.dvelx_y, out.dvelx_z,
        out.dvely_x, out.dvely_y, out.dvely_z,
        out.dvelz_x, out.dvelz_y, out.dvelz_z,
        out.dpress_x, out.dpress_y, out.dpress_z,
        grad_data;
        ndrange = n)
    synchronize && KA.synchronize(be)
    return out
end

function calculate_gradients_from_mesh_3d!(out::HydroGradients3D,
                                           mesh::ArepoMeshArrays3D,
                                           prim::PrimitiveState3D,
                                           center_x, center_y, center_z,
                                           face_center_x, face_center_y,
                                           face_center_z;
                                           box_size::Real = 1.0,
                                           gamma::Real = 5/3,
                                           synchronize::Bool = true)
    n = length(prim.rho)
    nf = length(mesh.c1)
    length(center_x) == n || error("center_x has wrong length")
    length(center_y) == n || error("center_y has wrong length")
    length(center_z) == n || error("center_z has wrong length")
    length(face_center_x) == nf || error("face_center_x has wrong length")
    length(face_center_y) == nf || error("face_center_y has wrong length")
    length(face_center_z) == nf || error("face_center_z has wrong length")
    be = KA.get_backend(out.drho_x)
    T = eltype(out.drho_x)
    cell_data = _backend_zeros(be, T, 9 * n)
    _pack_gradient_primitive_cell_data_3d_k!(be)(
        cell_data, prim.rho, prim.vx, prim.vy, prim.vz, prim.pressure,
        center_x, center_y, center_z, T(gamma);
        ndrange = n)
    face_data = _backend_zeros(be, T, 4 * nf)
    _pack_gradient_mesh_face_data_3d_k!(be)(
        face_data, mesh.face_area, face_center_x, face_center_y, face_center_z;
        ndrange = nf)
    grad_data = _backend_zeros(be, T, 15 * n)
    _gradients_from_mesh_3d_k!(be)(
        grad_data, mesh.cell_face_offsets, mesh.cell_faces,
        mesh.cell_face_signs, mesh.c1, mesh.c2, cell_data, face_data,
        T(box_size);
        ndrange = n)
    _unpack_gradients_3d_k!(be)(
        out.drho_x, out.drho_y, out.drho_z,
        out.dvelx_x, out.dvelx_y, out.dvelx_z,
        out.dvely_x, out.dvely_y, out.dvely_z,
        out.dvelz_x, out.dvelz_y, out.dvelz_z,
        out.dpress_x, out.dpress_y, out.dpress_z,
        grad_data;
        ndrange = n)
    synchronize && KA.synchronize(be)
    return out
end

function calculate_gradients_from_mesh_activecells_3d!(
    out::HydroGradients3D,
    mesh::ArepoMeshArrays3D,
    prim::PrimitiveState3D,
    center_x, center_y, center_z,
    face_center_x, face_center_y, face_center_z,
    active_counts, active_faces, active_signs;
    active_stride::Integer,
    box_size::Real = 1.0,
    gamma::Real = 5/3,
    synchronize::Bool = true)
    n = length(prim.rho)
    nf = length(mesh.c1)
    length(center_x) == n || error("center_x has wrong length")
    length(center_y) == n || error("center_y has wrong length")
    length(center_z) == n || error("center_z has wrong length")
    length(face_center_x) == nf || error("face_center_x has wrong length")
    length(face_center_y) == nf || error("face_center_y has wrong length")
    length(face_center_z) == nf || error("face_center_z has wrong length")
    length(active_counts) == n || error("active_counts has wrong length")
    length(active_faces) >= active_stride * n || error("active_faces has wrong length")
    length(active_signs) >= active_stride * n || error("active_signs has wrong length")
    be = KA.get_backend(out.drho_x)
    T = eltype(out.drho_x)
    cell_data = _backend_zeros(be, T, 9 * n)
    _pack_gradient_primitive_cell_data_3d_k!(be)(
        cell_data, prim.rho, prim.vx, prim.vy, prim.vz, prim.pressure,
        center_x, center_y, center_z, T(gamma);
        ndrange = n)
    face_data = _backend_zeros(be, T, 4 * nf)
    _pack_gradient_mesh_face_data_3d_k!(be)(
        face_data, mesh.face_area, face_center_x, face_center_y, face_center_z;
        ndrange = nf)
    grad_data = _backend_zeros(be, T, 15 * n)
    _gradients_from_mesh_activecells_3d_k!(be)(
        grad_data, active_counts, active_faces, active_signs,
        Int32(active_stride), mesh.c1, mesh.c2, cell_data, face_data,
        T(box_size);
        ndrange = n)
    _unpack_gradients_3d_k!(be)(
        out.drho_x, out.drho_y, out.drho_z,
        out.dvelx_x, out.dvelx_y, out.dvelx_z,
        out.dvely_x, out.dvely_y, out.dvely_z,
        out.dvelz_x, out.dvelz_y, out.dvelz_z,
        out.dpress_x, out.dpress_y, out.dpress_z,
        grad_data;
        ndrange = n)
    synchronize && KA.synchronize(be)
    return out
end

function _pack_hydro_gradients(::Type{T}, g::HydroGradients3D) where {T}
    return _pack_gradient_columns(T,
        Array(g.drho_x), Array(g.drho_y), Array(g.drho_z),
        Array(g.dvelx_x), Array(g.dvelx_y), Array(g.dvelx_z),
        Array(g.dvely_x), Array(g.dvely_y), Array(g.dvely_z),
        Array(g.dvelz_x), Array(g.dvelz_y), Array(g.dvelz_z),
        Array(g.dpress_x), Array(g.dpress_y), Array(g.dpress_z))
end

@inline function _predict_primitive_face_state3(cell::Int, f::Int, ncell::Int,
                                                nface::Int, cell_data,
                                                grad_data, face_data,
                                                box_size::T, gamma::T) where {T}
    r0 = cell_data[cell]
    vx0 = cell_data[ncell + cell]
    vy0 = cell_data[2 * ncell + cell]
    vz0 = cell_data[3 * ncell + cell]
    p0 = cell_data[4 * ncell + cell]
    cx = cell_data[5 * ncell + cell]
    cy = cell_data[6 * ncell + cell]
    cz = cell_data[7 * ncell + cell]
    dt = cell_data[8 * ncell + cell]

    grx = grad_data[cell]
    gry = grad_data[ncell + cell]
    grz = grad_data[2 * ncell + cell]
    gvxx = grad_data[3 * ncell + cell]
    gvxy = grad_data[4 * ncell + cell]
    gvxz = grad_data[5 * ncell + cell]
    gvyx = grad_data[6 * ncell + cell]
    gvyy = grad_data[7 * ncell + cell]
    gvyz = grad_data[8 * ncell + cell]
    gvzx = grad_data[9 * ncell + cell]
    gvzy = grad_data[10 * ncell + cell]
    gvzz = grad_data[11 * ncell + cell]
    gpx = grad_data[12 * ncell + cell]
    gpy = grad_data[13 * ncell + cell]
    gpz = grad_data[14 * ncell + cell]

    fx = face_data[f]
    fy = face_data[nface + f]
    fz = face_data[2 * nface + f]
    wx = face_data[3 * nface + f]
    wy = face_data[4 * nface + f]
    wz = face_data[5 * nface + f]

    dx = _periodic_delta(fx - cx, box_size)
    dy = _periodic_delta(fy - cy, box_size)
    dz = _periodic_delta(fz - cz, box_size)

    vx = vx0 - wx
    vy = vy0 - wy
    vz = vz0 - wz

    dr_time = -dt * (vx * grx + r0 * gvxx + vy * gry + r0 * gvyy +
                     vz * grz + r0 * gvzz)
    dvx_time = -dt * (gpx / r0 + vx * gvxx + vy * gvxy + vz * gvxz)
    dvy_time = -dt * (gpy / r0 + vx * gvyx + vy * gvyy + vz * gvyz)
    dvz_time = -dt * (gpz / r0 + vx * gvzx + vy * gvzy + vz * gvzz)
    dp_time = -dt * (gamma * p0 * (gvxx + gvyy + gvzz) +
                     vx * gpx + vy * gpy + vz * gpz)

    dr_space = grx * dx + gry * dy + grz * dz
    dvx_space = gvxx * dx + gvxy * dy + gvxz * dz
    dvy_space = gvyx * dx + gvyy * dy + gvyz * dz
    dvz_space = gvzx * dx + gvzy * dy + gvzz * dz
    dp_space = gpx * dx + gpy * dy + gpz * dz

    r = r0
    p = p0
    if r0 > zero(T) && r0 + dr_time + dr_space >= zero(T) &&
       p0 + dp_time + dp_space >= zero(T)
        r += dr_time + dr_space
        vx += dvx_time + dvx_space
        vy += dvy_time + dvy_space
        vz += dvz_time + dvz_space
        p += dp_time + dp_space
    end
    return (r, vx, vy, vz, p)
end

@kernel function _predict_face_states_3d_k!(
    left, right, @Const(c1), @Const(c2),
    @Const(cell_data), @Const(grad_data), @Const(face_data),
    box_size, gamma)
    f = @index(Global, Linear)
    T = eltype(left)
    ncell = Int(length(cell_data) ÷ 9)
    nface = Int(length(left) ÷ 5)
    @inbounds begin
        i = Int(c1[f])
        j = Int(c2[f])
        L = _predict_primitive_face_state3(i, f, ncell, nface, cell_data,
                                           grad_data, face_data, box_size, gamma)
        left[f] = L[1]
        left[nface + f] = L[2]
        left[2 * nface + f] = L[3]
        left[3 * nface + f] = L[4]
        left[4 * nface + f] = L[5]
        if j > 0
            R = _predict_primitive_face_state3(j, f, ncell, nface, cell_data,
                                               grad_data, face_data, box_size, gamma)
            right[f] = R[1]
            right[nface + f] = R[2]
            right[2 * nface + f] = R[3]
            right[3 * nface + f] = R[4]
            right[4 * nface + f] = R[5]
        else
            right[f] = L[1]
            right[nface + f] = L[2]
            right[2 * nface + f] = L[3]
            right[3 * nface + f] = L[4]
            right[4 * nface + f] = L[5]
        end
    end
end

function predict_face_states_3d!(states::FaceStates3D, mesh::ArepoMeshArrays3D,
                                 gradients::HydroGradients3D,
                                 rho, velx, vely, velz, pressure, center,
                                 face_center;
                                 dt_extrapolation = nothing,
                                 box_size::Real = 1.0,
                                 gamma::Real = 5/3)
    n = length(rho)
    nf = length(mesh.c1)
    size(center) == (n, 3) || error("center must be n x 3")
    size(face_center) == (nf, 3) || error("face_center must be nf x 3")
    length(states.left) == 5nf || error("left face-state buffer has wrong length")
    length(states.right) == 5nf || error("right face-state buffer has wrong length")
    be = KA.get_backend(states.left)
    T = eltype(states.left)
    I = eltype(mesh.c1)
    dt = dt_extrapolation === nothing ? zeros(T, n) : dt_extrapolation
    c1 = _backend_copy(be, Array(mesh.c1), I)
    c2 = _backend_copy(be, Array(mesh.c2), I)
    cell_data = _backend_copy(be,
        _pack_gradient_columns(T, rho, velx, vely, velz, pressure,
                               view(center, :, 1), view(center, :, 2),
                               view(center, :, 3), dt), T)
    grad_data = _pack_hydro_gradients_backend(be, T, gradients)
    face_data = _backend_copy(be,
        _pack_gradient_columns(T, view(face_center, :, 1), view(face_center, :, 2),
                               view(face_center, :, 3), Array(mesh.face_vx),
                               Array(mesh.face_vy), Array(mesh.face_vz),
                               Array(mesh.normal_x), Array(mesh.normal_y),
                               Array(mesh.normal_z)), T)
    _predict_face_states_3d_k!(be)(states.left, states.right, c1, c2,
                                   cell_data, grad_data, face_data,
                                   T(box_size), T(gamma);
                                   ndrange = nf)
    KA.synchronize(be)
    return states
end

function predict_face_states_3d!(states::FaceStates3D, mesh::ArepoMeshArrays3D,
                                 gradients::HydroGradients3D,
                                 prim::PrimitiveState3D, center, face_center;
                                 dt_extrapolation = nothing,
                                 box_size::Real = 1.0,
                                 gamma::Real = 5/3,
                                 synchronize::Bool = true)
    n = length(prim.rho)
    nf = length(mesh.c1)
    size(center) == (n, 3) || error("center must be n x 3")
    size(face_center) == (nf, 3) || error("face_center must be nf x 3")
    length(states.left) == 5nf || error("left face-state buffer has wrong length")
    length(states.right) == 5nf || error("right face-state buffer has wrong length")
    be = KA.get_backend(states.left)
    T = eltype(states.left)
    dt = dt_extrapolation === nothing ?
         _backend_copy(be, fill(zero(T), n), T) :
         _backend_copy(be, Array(dt_extrapolation), T)
    cx = _backend_copy(be, collect(view(center, :, 1)), T)
    cy = _backend_copy(be, collect(view(center, :, 2)), T)
    cz = _backend_copy(be, collect(view(center, :, 3)), T)
    cell_data = _backend_zeros(be, T, 9 * n)
    _pack_predict_primitive_cell_data_3d_k!(be)(
        cell_data, prim.rho, prim.vx, prim.vy, prim.vz, prim.pressure,
        cx, cy, cz, dt;
        ndrange = n)
    grad_data = _pack_hydro_gradients_backend(be, T, gradients)
    fcx = _backend_copy(be, collect(view(face_center, :, 1)), T)
    fcy = _backend_copy(be, collect(view(face_center, :, 2)), T)
    fcz = _backend_copy(be, collect(view(face_center, :, 3)), T)
    face_data = _backend_zeros(be, T, 9 * nf)
    _pack_predict_face_data_3d_k!(be)(
        face_data, fcx, fcy, fcz, mesh.face_vx, mesh.face_vy, mesh.face_vz,
        mesh.normal_x, mesh.normal_y, mesh.normal_z;
        ndrange = nf)
    _predict_face_states_3d_k!(be)(states.left, states.right, mesh.c1, mesh.c2,
                                   cell_data, grad_data, face_data,
                                   T(box_size), T(gamma);
                                   ndrange = nf)
    synchronize && KA.synchronize(be)
    return states
end

function predict_face_states_3d!(states::FaceStates3D, mesh::ArepoMeshArrays3D,
                                 gradients::HydroGradients3D,
                                 prim::PrimitiveState3D,
                                 center_x, center_y, center_z,
                                 face_center_x, face_center_y, face_center_z;
                                 dt_extrapolation = nothing,
                                 box_size::Real = 1.0,
                                 gamma::Real = 5/3,
                                 synchronize::Bool = true)
    n = length(prim.rho)
    nf = length(mesh.c1)
    length(center_x) == n || error("center_x has wrong length")
    length(center_y) == n || error("center_y has wrong length")
    length(center_z) == n || error("center_z has wrong length")
    length(face_center_x) == nf || error("face_center_x has wrong length")
    length(face_center_y) == nf || error("face_center_y has wrong length")
    length(face_center_z) == nf || error("face_center_z has wrong length")
    length(states.left) == 5nf || error("left face-state buffer has wrong length")
    length(states.right) == 5nf || error("right face-state buffer has wrong length")
    be = KA.get_backend(states.left)
    T = eltype(states.left)
    dt = dt_extrapolation === nothing ?
         _backend_copy(be, fill(zero(T), n), T) :
         dt_extrapolation
    cell_data = _backend_zeros(be, T, 9 * n)
    _pack_predict_primitive_cell_data_3d_k!(be)(
        cell_data, prim.rho, prim.vx, prim.vy, prim.vz, prim.pressure,
        center_x, center_y, center_z, dt;
        ndrange = n)
    grad_data = _pack_hydro_gradients_backend(be, T, gradients)
    face_data = _backend_zeros(be, T, 9 * nf)
    _pack_predict_face_data_3d_k!(be)(
        face_data, face_center_x, face_center_y, face_center_z,
        mesh.face_vx, mesh.face_vy, mesh.face_vz,
        mesh.normal_x, mesh.normal_y, mesh.normal_z;
        ndrange = nf)
    _predict_face_states_3d_k!(be)(states.left, states.right, mesh.c1, mesh.c2,
                                   cell_data, grad_data, face_data,
                                   T(box_size), T(gamma);
                                   ndrange = nf)
    synchronize && KA.synchronize(be)
    return states
end

struct _Plane3D
    a1::Float64
    a2::Float64
    a3::Float64
    b::Float64
    neighbor::Int
end

function _domain3(domain)
    return ((float(domain[1][1]), float(domain[1][2])),
            (float(domain[2][1]), float(domain[2][2])),
            (float(domain[3][1]), float(domain[3][2])))
end

function _bounded_voronoi_planes3(points, i::Int, domain)
    pix, piy, piz = points[i, 1], points[i, 2], points[i, 3]
    planes = _Plane3D[
        _Plane3D( 1.0,  0.0,  0.0, domain[1][2], 0),
        _Plane3D(-1.0,  0.0,  0.0, -domain[1][1], 0),
        _Plane3D( 0.0,  1.0,  0.0, domain[2][2], 0),
        _Plane3D( 0.0, -1.0,  0.0, -domain[2][1], 0),
        _Plane3D( 0.0,  0.0,  1.0, domain[3][2], 0),
        _Plane3D( 0.0,  0.0, -1.0, -domain[3][1], 0),
    ]
    n = size(points, 1)
    @inbounds for j in 1:n
        i == j && continue
        pjx, pjy, pjz = points[j, 1], points[j, 2], points[j, 3]
        a1 = 2 * (pjx - pix)
        a2 = 2 * (pjy - piy)
        a3 = 2 * (pjz - piz)
        b = pjx * pjx + pjy * pjy + pjz * pjz - pix * pix - piy * piy - piz * piz
        push!(planes, _Plane3D(a1, a2, a3, b, j))
    end
    return planes
end

function _periodic_voronoi_planes3(points, i::Int, domain)
    pix, piy, piz = points[i, 1], points[i, 2], points[i, 3]
    lx = domain[1][2] - domain[1][1]
    ly = domain[2][2] - domain[2][1]
    lz = domain[3][2] - domain[3][1]
    planes = _Plane3D[
        _Plane3D( 1.0,  0.0,  0.0, pix + 0.5 * lx, 0),
        _Plane3D(-1.0,  0.0,  0.0, -(pix - 0.5 * lx), 0),
        _Plane3D( 0.0,  1.0,  0.0, piy + 0.5 * ly, 0),
        _Plane3D( 0.0, -1.0,  0.0, -(piy - 0.5 * ly), 0),
        _Plane3D( 0.0,  0.0,  1.0, piz + 0.5 * lz, 0),
        _Plane3D( 0.0,  0.0, -1.0, -(piz - 0.5 * lz), 0),
    ]
    n = size(points, 1)
    @inbounds for j in 1:n
        for sz in -1:1, sy in -1:1, sx in -1:1
            j == i && sx == 0 && sy == 0 && sz == 0 && continue
            pjx = points[j, 1] + sx * lx
            pjy = points[j, 2] + sy * ly
            pjz = points[j, 3] + sz * lz
            a1 = 2 * (pjx - pix)
            a2 = 2 * (pjy - piy)
            a3 = 2 * (pjz - piz)
            b = pjx * pjx + pjy * pjy + pjz * pjz - pix * pix - piy * piy - piz * piz
            push!(planes, _Plane3D(a1, a2, a3, b, j == i ? 0 : j))
        end
    end
    return planes
end

function _wrap_bin3(raw::Int, nb::Int)
    q = fld(raw - 1, nb)
    return raw - q * nb, q
end

_bin_id3(ix::Int, iy::Int, iz::Int, nb::Int) = ix + nb * (iy - 1) + nb * nb * (iz - 1)

function _periodic_bins3(points, domain, bins_per_axis::Int)
    nb = bins_per_axis
    bins = [Int[] for _ in 1:(nb^3)]
    coords = Matrix{Int}(undef, size(points, 1), 3)
    lx = domain[1][2] - domain[1][1]
    ly = domain[2][2] - domain[2][1]
    lz = domain[3][2] - domain[3][1]
    @inbounds for i in axes(points, 1)
        ix = clamp(floor(Int, nb * (points[i, 1] - domain[1][1]) / lx) + 1, 1, nb)
        iy = clamp(floor(Int, nb * (points[i, 2] - domain[2][1]) / ly) + 1, 1, nb)
        iz = clamp(floor(Int, nb * (points[i, 3] - domain[3][1]) / lz) + 1, 1, nb)
        coords[i, 1] = ix
        coords[i, 2] = iy
        coords[i, 3] = iz
        push!(bins[_bin_id3(ix, iy, iz, nb)], i)
    end
    return bins, coords
end

function _periodic_local_voronoi_planes3(points, i::Int, domain, bins, coords;
                                         search_radius::Int)
    pix, piy, piz = points[i, 1], points[i, 2], points[i, 3]
    lx = domain[1][2] - domain[1][1]
    ly = domain[2][2] - domain[2][1]
    lz = domain[3][2] - domain[3][1]
    nb = round(Int, cbrt(length(bins)))
    planes = _Plane3D[
        _Plane3D( 1.0,  0.0,  0.0, pix + 0.5 * lx, 0),
        _Plane3D(-1.0,  0.0,  0.0, -(pix - 0.5 * lx), 0),
        _Plane3D( 0.0,  1.0,  0.0, piy + 0.5 * ly, 0),
        _Plane3D( 0.0, -1.0,  0.0, -(piy - 0.5 * ly), 0),
        _Plane3D( 0.0,  0.0,  1.0, piz + 0.5 * lz, 0),
        _Plane3D( 0.0,  0.0, -1.0, -(piz - 0.5 * lz), 0),
    ]
    bix, biy, biz = coords[i, 1], coords[i, 2], coords[i, 3]
    @inbounds for oz in -search_radius:search_radius,
                  oy in -search_radius:search_radius,
                  ox in -search_radius:search_radius
        bx, sx = _wrap_bin3(bix + ox, nb)
        by, sy = _wrap_bin3(biy + oy, nb)
        bz, sz = _wrap_bin3(biz + oz, nb)
        for j in bins[_bin_id3(bx, by, bz, nb)]
            j == i && sx == 0 && sy == 0 && sz == 0 && continue
            pjx = points[j, 1] + sx * lx
            pjy = points[j, 2] + sy * ly
            pjz = points[j, 3] + sz * lz
            a1 = 2 * (pjx - pix)
            a2 = 2 * (pjy - piy)
            a3 = 2 * (pjz - piz)
            b = pjx * pjx + pjy * pjy + pjz * pjz - pix * pix - piy * piy - piz * piz
            push!(planes, _Plane3D(a1, a2, a3, b, j == i ? 0 : j))
        end
    end
    return planes
end

@inline function _det3(a11, a12, a13, a21, a22, a23, a31, a32, a33)
    return a11 * (a22 * a33 - a23 * a32) -
           a12 * (a21 * a33 - a23 * a31) +
           a13 * (a21 * a32 - a22 * a31)
end

function _intersect_planes3(p1::_Plane3D, p2::_Plane3D, p3::_Plane3D;
                            tol::Float64)
    det = _det3(p1.a1, p1.a2, p1.a3,
                p2.a1, p2.a2, p2.a3,
                p3.a1, p3.a2, p3.a3)
    abs(det) <= tol && return nothing
    x = _det3(p1.b, p1.a2, p1.a3,
              p2.b, p2.a2, p2.a3,
              p3.b, p3.a2, p3.a3) / det
    y = _det3(p1.a1, p1.b, p1.a3,
              p2.a1, p2.b, p2.a3,
              p3.a1, p3.b, p3.a3) / det
    z = _det3(p1.a1, p1.a2, p1.b,
              p2.a1, p2.a2, p2.b,
              p3.a1, p3.a2, p3.b) / det
    return (x, y, z)
end

function _inside_planes3(x, y, z, planes; tol::Float64)
    @inbounds for p in planes
        p.a1 * x + p.a2 * y + p.a3 * z <= p.b + tol || return false
    end
    return true
end

function _push_unique_vertex3!(verts::Vector{NTuple{3,Float64}}, v; tol::Float64)
    x, y, z = v
    @inbounds for u in verts
        dx = x - u[1]
        dy = y - u[2]
        dz = z - u[3]
        dx * dx + dy * dy + dz * dz <= tol * tol && return
    end
    push!(verts, (x, y, z))
    return
end

function _cell_vertices3(planes; tol::Float64)
    verts = NTuple{3,Float64}[]
    m = length(planes)
    @inbounds for a in 1:(m - 2), b in (a + 1):(m - 1), c in (b + 1):m
        v = _intersect_planes3(planes[a], planes[b], planes[c]; tol)
        v === nothing && continue
        _inside_planes3(v[1], v[2], v[3], planes; tol = 10tol) || continue
        _push_unique_vertex3!(verts, v; tol = 100tol)
    end
    return verts
end

function _vertices_matrix3(verts::Vector{NTuple{3,Float64}})
    out = Matrix{Float64}(undef, length(verts), 3)
    @inbounds for i in eachindex(verts)
        out[i, 1] = verts[i][1]
        out[i, 2] = verts[i][2]
        out[i, 3] = verts[i][3]
    end
    return out
end

function _ring_area_normal3(verts::Vector{NTuple{3,Float64}})
    sx = 0.0; sy = 0.0; sz = 0.0
    nv = length(verts)
    @inbounds for i in 1:nv
        j = i == nv ? 1 : i + 1
        ax, ay, az = verts[i]
        bx, by, bz = verts[j]
        sx += ay * bz - az * by
        sy += az * bx - ax * bz
        sz += ax * by - ay * bx
    end
    return (sx, sy, sz)
end

function _centroid3(verts::Vector{NTuple{3,Float64}})
    sx = 0.0; sy = 0.0; sz = 0.0
    @inbounds for v in verts
        sx += v[1]
        sy += v[2]
        sz += v[3]
    end
    invn = 1.0 / length(verts)
    return (sx * invn, sy * invn, sz * invn)
end

function _order_face_vertices3(verts::Vector{NTuple{3,Float64}}, plane::_Plane3D)
    n = length(verts)
    n < 3 && return verts
    cx, cy, cz = _centroid3(verts)
    nx, ny, nz = plane.a1, plane.a2, plane.a3
    nn = sqrt(nx * nx + ny * ny + nz * nz)
    nx /= nn; ny /= nn; nz /= nn
    ux, uy, uz = abs(nx) < 0.8 ? (0.0, -nz, ny) : (-nz, 0.0, nx)
    un = sqrt(ux * ux + uy * uy + uz * uz)
    ux /= un; uy /= un; uz /= un
    vx = ny * uz - nz * uy
    vy = nz * ux - nx * uz
    vz = nx * uy - ny * ux
    idx = collect(1:n)
    sort!(idx; by = q -> begin
        px = verts[q][1] - cx
        py = verts[q][2] - cy
        pz = verts[q][3] - cz
        atan(px * vx + py * vy + pz * vz, px * ux + py * uy + pz * uz)
    end)
    ordered = verts[idx]
    sx, sy, sz = _ring_area_normal3(ordered)
    if sx * plane.a1 + sy * plane.a2 + sz * plane.a3 < 0
        reverse!(ordered)
    end
    return ordered
end

function _bounded_voronoi_cell3(points, i::Int, domain; tol::Float64)
    planes = _bounded_voronoi_planes3(points, i, domain)
    return _voronoi_cell_from_planes3(planes; tol)
end

function _periodic_voronoi_cell3(points, i::Int, domain; tol::Float64)
    planes = _periodic_voronoi_planes3(points, i, domain)
    return _voronoi_cell_from_planes3(planes; tol)
end

function _periodic_local_voronoi_cell3(points, i::Int, domain, bins, coords;
                                       search_radius::Int, tol::Float64)
    planes = _periodic_local_voronoi_planes3(points, i, domain, bins, coords;
                                             search_radius)
    return _voronoi_cell_from_planes3(planes; tol)
end

function _voronoi_cell_from_planes3(planes; tol::Float64)
    verts = _cell_vertices3(planes; tol)
    faces = Vector{NamedTuple}()
    volume = 0.0
    @inbounds for plane in planes
        fverts = NTuple{3,Float64}[]
        for v in verts
            if abs(plane.a1 * v[1] + plane.a2 * v[2] + plane.a3 * v[3] - plane.b) <= 100tol
                push!(fverts, v)
            end
        end
        length(fverts) >= 3 || continue
        fverts = _order_face_vertices3(fverts, plane)
        sx, sy, sz = _ring_area_normal3(fverts)
        area = 0.5 * sqrt(sx * sx + sy * sy + sz * sz)
        area > 100tol || continue
        fx, fy, fz = _centroid3(fverts)
        volume += (fx * (0.5 * sx) + fy * (0.5 * sy) + fz * (0.5 * sz)) / 3
        push!(faces, (; neighbor = plane.neighbor, center = (fx, fy, fz), area,
                      normal = (plane.a1, plane.a2, plane.a3)))
    end
    c = isempty(verts) ? (NaN, NaN, NaN) :
        _centroid3(verts)
    return (; vertices = verts, faces, volume = abs(volume), center = c)
end

"""
    bounded_voronoi_mesh_arrays_3d(points; domain=((0,1),(0,1),(0,1)), ...)

Build a bounded 3-D Voronoi face table by clipping each cell against all
pairwise bisectors and the domain box. This is a small-mesh correctness and
rebuild gate, not the production AREPO-scale tessellator.

Returns `(; geom, volume, center, face_center, generators, domain)`.
"""
function bounded_voronoi_mesh_arrays_3d(points::AbstractMatrix;
                                        domain = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                                        T::Type{<:AbstractFloat} = Float64,
                                        index_type::Type{<:Integer} = Int32,
                                        face_velocity = nothing,
                                        cell_velocity = nothing,
                                        tol::Float64 = 1e-10)
    size(points, 2) == 3 || error("points must be n x 3")
    pts = Matrix{Float64}(points)
    n = size(pts, 1)
    dom = _domain3(domain)
    cells = [_bounded_voronoi_cell3(pts, i, dom; tol) for i in 1:n]
    volume = [cells[i].volume for i in 1:n]
    center = Matrix{Float64}(undef, n, 3)
    for i in 1:n
        center[i, 1] = cells[i].center[1]
        center[i, 2] = cells[i].center[2]
        center[i, 3] = cells[i].center[3]
    end
    c1 = Int[]
    c2 = Int[]
    area = Float64[]
    normals = Matrix{Float64}(undef, 0, 3)
    face_center = Matrix{Float64}(undef, 0, 3)
    @inbounds for i in 1:n
        for face in cells[i].faces
            j = face.neighbor
            j > 0 && i > j && continue
            nx, ny, nz = face.normal
            nn = sqrt(nx * nx + ny * ny + nz * nz)
            push!(c1, i)
            push!(c2, j)
            push!(area, face.area)
            normals = vcat(normals, reshape([nx / nn, ny / nn, nz / nn], 1, 3))
            face_center = vcat(face_center, reshape(collect(face.center), 1, 3))
        end
    end
    fvx, fvy, fvz = _face_velocity_arrays_3d(c1, c2, face_velocity, cell_velocity, T)
    offsets, faces, signs = _cell_face_csr(n, c1, c2, index_type)
    geom = ArepoMeshArrays3D(index_type.(c1), index_type.(c2), offsets, faces, signs,
                             T.(volume), T.(area), T.(normals[:, 1]),
                             T.(normals[:, 2]), T.(normals[:, 3]), fvx, fvy, fvz)
    return (; geom, volume, center, face_center, generators = pts, domain = dom)
end

"""
    periodic_voronoi_mesh_arrays_3d(points; domain=((0,1),(0,1),(0,1)), ...)

Build a small periodic 3-D Voronoi face table by clipping each cell in a local
periodic image cube. This is still an all-pairs diagnostic producer, but it
removes the bounded-domain approximation and keeps periodic duplicate faces.
"""
function periodic_voronoi_mesh_arrays_3d(points::AbstractMatrix;
                                         domain = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                                         T::Type{<:AbstractFloat} = Float64,
                                         index_type::Type{<:Integer} = Int32,
                                         face_velocity = nothing,
                                         cell_velocity = nothing,
                                         tol::Float64 = 1e-10)
    size(points, 2) == 3 || error("points must be n x 3")
    pts = Matrix{Float64}(points)
    n = size(pts, 1)
    dom = _domain3(domain)
    cells = [_periodic_voronoi_cell3(pts, i, dom; tol) for i in 1:n]
    volume = [cells[i].volume for i in 1:n]
    center = Matrix{Float64}(undef, n, 3)
    for i in 1:n
        center[i, 1] = cells[i].center[1]
        center[i, 2] = cells[i].center[2]
        center[i, 3] = cells[i].center[3]
    end
    c1 = Int[]
    c2 = Int[]
    area = Float64[]
    normals = Matrix{Float64}(undef, 0, 3)
    face_center = Matrix{Float64}(undef, 0, 3)
    @inbounds for i in 1:n
        for face in cells[i].faces
            j = face.neighbor
            (j <= 0 || i > j) && continue
            nx, ny, nz = face.normal
            nn = sqrt(nx * nx + ny * ny + nz * nz)
            push!(c1, i)
            push!(c2, j)
            push!(area, face.area)
            normals = vcat(normals, reshape([nx / nn, ny / nn, nz / nn], 1, 3))
            face_center = vcat(face_center, reshape(collect(face.center), 1, 3))
        end
    end
    fvx, fvy, fvz = _face_velocity_arrays_3d(c1, c2, face_velocity, cell_velocity, T)
    offsets, faces, signs = _cell_face_csr(n, c1, c2, index_type)
    geom = ArepoMeshArrays3D(index_type.(c1), index_type.(c2), offsets, faces, signs,
                             T.(volume), T.(area), T.(normals[:, 1]),
                             T.(normals[:, 2]), T.(normals[:, 3]), fvx, fvy, fvz)
    return (; geom, volume, center, face_center, generators = pts, domain = dom)
end

"""
    local_periodic_voronoi_mesh_arrays_3d(points; bins_per_axis=nothing,
                                          search_radius=1, ...)

Build a periodic 3-D Voronoi face table using only generator images in a local
periodic bin stencil. This scales the native rebuild gate to the current
near-lattice turbulence boxes without adding an external Delaunay dependency.
It is a production-path rung, not a mathematical replacement for a full
Delaunay-backed tessellator on arbitrary point sets.
"""
function local_periodic_voronoi_mesh_arrays_3d(points::AbstractMatrix;
                                               domain = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                                               T::Type{<:AbstractFloat} = Float64,
                                               index_type::Type{<:Integer} = Int32,
                                               face_velocity = nothing,
                                               cell_velocity = nothing,
                                               bins_per_axis = nothing,
                                               search_radius::Integer = 1,
                                               threaded::Bool = Threads.nthreads() > 1,
                                               tol::Float64 = 1e-10)
    size(points, 2) == 3 || error("points must be n x 3")
    pts = Matrix{Float64}(points)
    n = size(pts, 1)
    dom = _domain3(domain)
    nb = bins_per_axis === nothing ? max(1, round(Int, cbrt(n))) : Int(bins_per_axis)
    nb > 0 || error("bins_per_axis must be positive")
    search_radius >= 1 || error("search_radius must be at least 1")
    bins, coords = _periodic_bins3(pts, dom, nb)
    first_cell = _periodic_local_voronoi_cell3(pts, 1, dom, bins, coords;
                                               search_radius = Int(search_radius), tol)
    cells = Vector{typeof(first_cell)}(undef, n)
    cells[1] = first_cell
    if threaded && n > 1
        Threads.@threads for i in 2:n
            cells[i] = _periodic_local_voronoi_cell3(pts, i, dom, bins, coords;
                                                     search_radius = Int(search_radius), tol)
        end
    else
        @inbounds for i in 2:n
            cells[i] = _periodic_local_voronoi_cell3(pts, i, dom, bins, coords;
                                                     search_radius = Int(search_radius), tol)
        end
    end
    volume = [cells[i].volume for i in 1:n]
    center = Matrix{Float64}(undef, n, 3)
    for i in 1:n
        center[i, 1] = cells[i].center[1]
        center[i, 2] = cells[i].center[2]
        center[i, 3] = cells[i].center[3]
    end
    c1 = Int[]
    c2 = Int[]
    area = Float64[]
    normal_tuples = NTuple{3,Float64}[]
    face_center_tuples = NTuple{3,Float64}[]
    @inbounds for i in 1:n
        for face in cells[i].faces
            j = face.neighbor
            (j <= 0 || i > j) && continue
            nx, ny, nz = face.normal
            nn = sqrt(nx * nx + ny * ny + nz * nz)
            push!(c1, i)
            push!(c2, j)
            push!(area, face.area)
            push!(normal_tuples, (nx / nn, ny / nn, nz / nn))
            push!(face_center_tuples, face.center)
        end
    end
    nf = length(c1)
    normals = Matrix{Float64}(undef, nf, 3)
    face_center = Matrix{Float64}(undef, nf, 3)
    @inbounds for f in 1:nf
        normals[f, 1] = normal_tuples[f][1]
        normals[f, 2] = normal_tuples[f][2]
        normals[f, 3] = normal_tuples[f][3]
        face_center[f, 1] = face_center_tuples[f][1]
        face_center[f, 2] = face_center_tuples[f][2]
        face_center[f, 3] = face_center_tuples[f][3]
    end
    fvx, fvy, fvz = _face_velocity_arrays_3d(c1, c2, face_velocity, cell_velocity, T)
    offsets, faces, signs = _cell_face_csr(n, c1, c2, index_type)
    geom = ArepoMeshArrays3D(index_type.(c1), index_type.(c2), offsets, faces, signs,
                             T.(volume), T.(area), T.(normals[:, 1]),
                             T.(normals[:, 2]), T.(normals[:, 3]), fvx, fvy, fvz)
    return (; geom, volume, center, face_center, generators = pts, domain = dom,
            bins_per_axis = nb, search_radius = Int(search_radius))
end

function advect_generators_3d(points::AbstractMatrix, velocity, dt::Real,
                              domain = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0));
                              boundary::Symbol = :clamp)
    pts = Matrix{Float64}(points)
    size(pts, 2) == 3 || error("points must be n x 3")
    v = _velocity_matrix3(velocity, size(pts, 1))
    out = pts .+ float(dt) .* v
    if boundary == :clamp
        dom = _domain3(domain)
        @inbounds for d in 1:3
            lo, hi = dom[d]
            out[:, d] .= clamp.(@view(out[:, d]), lo, hi)
        end
    elseif boundary == :periodic
        dom = _domain3(domain)
        @inbounds for d in 1:3
            lo, hi = dom[d]
            len = hi - lo
            out[:, d] .= lo .+ mod.(@view(out[:, d]) .- lo, len)
        end
    elseif boundary == :none
    else
        error("unknown moving mesh boundary=$boundary; use :clamp, :periodic, or :none")
    end
    return out
end

function _mesh_velocity_from_input3(state::EulerState3D, mesh_velocity;
                                    gamma::Real)
    n = length(state.D)
    if mesh_velocity === nothing
        prim = conserved_to_primitive_3d(state; gamma)
        return hcat(prim.vx, prim.vy, prim.vz)
    end
    return _velocity_matrix3(mesh_velocity, n)
end

function moving_mesh_step_3d!(state::EulerState3D, points::AbstractMatrix;
                              dt::Real, gamma::Real, mesh_velocity = nothing,
                              domain = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                              boundary::Symbol = :clamp,
                              rebuild::Symbol = :auto,
                              local_bins_per_axis = nothing,
                              local_search_radius::Integer = 1,
                              riemann::Symbol = :hll,
                              T::Type{<:AbstractFloat} = Float64,
                              index_type::Type{<:Integer} = Int32)
    be = KA.get_backend(state.D)
    vmesh = _mesh_velocity_from_input3(state, mesh_velocity; gamma)
    npoints = size(points, 1)
    if boundary == :periodic && (rebuild == :local ||
                                 (rebuild == :auto && npoints > 128))
        old = local_periodic_voronoi_mesh_arrays_3d(points; domain, T = Float64,
                                                    index_type, cell_velocity = vmesh,
                                                    bins_per_axis = local_bins_per_axis,
                                                    search_radius = local_search_radius)
        new_points = advect_generators_3d(points, vmesh, dt, domain; boundary)
        new = local_periodic_voronoi_mesh_arrays_3d(new_points; domain, T = Float64,
                                                    index_type,
                                                    bins_per_axis = old.bins_per_axis,
                                                    search_radius = local_search_radius)
    else
        builder = boundary == :periodic ? periodic_voronoi_mesh_arrays_3d :
                  bounded_voronoi_mesh_arrays_3d
        old = builder(points; domain, T = Float64, index_type,
                      cell_velocity = vmesh)
        new_points = advect_generators_3d(points, vmesh, dt, domain; boundary)
        new = builder(new_points; domain, T = Float64, index_type)
    end
    old_geom = to_backend(be, old.geom; T, index_type)
    new_volume = _backend_copy(be, new.volume, T)
    finite_volume_step_3d!(state, old_geom; dt, gamma, riemann,
                           new_volume)
    new_geom = to_backend(be, new.geom; T, index_type)
    return (; points = new_points, geom = new_geom, state, mesh_velocity = vmesh,
            center = new.center, face_center = new.face_center)
end
