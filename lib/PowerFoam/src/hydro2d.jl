# AREPO-shaped 2-D Euler finite-volume kernels.
#
# This is deliberately a first-order face-table solver.  The important seam is
# the data flow: one kernel computes fluxes on AREPO/Voronoi faces, a second
# kernel gathers incident face fluxes per cell through a CSR table.  That is the
# same split we need for GPU execution without atomics.

struct ArepoMeshArrays2D{I<:AbstractVector,R<:AbstractVector,S<:AbstractVector}
    c1::I
    c2::I
    cell_face_offsets::I
    cell_faces::I
    cell_face_signs::S
    volume::R
    face_area::R
    normal_x::R
    normal_y::R
    face_vx::R
    face_vy::R
end

struct EulerState2D{A<:AbstractVector}
    D::A
    Mx::A
    My::A
    E::A
end

struct PrimitiveState2D{A<:AbstractVector}
    rho::A
    vx::A
    vy::A
    pressure::A
end

struct FaceFluxWork2D{A<:AbstractVector}
    FD::A
    FMx::A
    FMy::A
    FE::A
end

struct FaceStates2D{A<:AbstractVector}
    left::A
    right::A
end

struct HydroGradients2D{A<:AbstractVector}
    drho_x::A
    drho_y::A
    dvelx_x::A
    dvelx_y::A
    dvely_x::A
    dvely_y::A
    dpress_x::A
    dpress_y::A
end

function _cell_face_csr(mesh::PolygonMesh2D, ::Type{I}) where {I<:Integer}
    nc = length(mesh.cells)
    counts = zeros(Int, nc)
    for f in eachindex(mesh.faces.c1)
        counts[mesh.faces.c1[f]] += 1
        j = mesh.faces.c2[f]
        j > 0 && (counts[j] += 1)
    end
    offsets = Vector{I}(undef, nc + 1)
    offsets[1] = one(I)
    for i in 1:nc
        offsets[i + 1] = offsets[i] + I(counts[i])
    end
    faces = Vector{I}(undef, Int(offsets[end] - one(I)))
    signs = Vector{I}(undef, length(faces))
    cursor = Int.(offsets[1:end-1])
    for f in eachindex(mesh.faces.c1)
        i = mesh.faces.c1[f]
        p = cursor[i]
        faces[p] = I(f)
        signs[p] = -one(I)
        cursor[i] += 1
        j = mesh.faces.c2[f]
        if j > 0
            p = cursor[j]
            faces[p] = I(f)
            signs[p] = one(I)
            cursor[j] += 1
        end
    end
    return offsets, faces, signs
end

"""
    arepo_mesh_arrays(mesh; T=Float64, index_type=Int32)

Extract a flat AREPO-like mesh table from `PolygonMesh2D`: cell volumes (areas),
face owners/neighbors, face lengths, face normals, and a cell-to-face CSR table.
The returned arrays are ordinary host arrays; call [`to_backend`](@ref) to stage
them on a KernelAbstractions backend.
"""
function _face_velocity_arrays(mesh::PolygonMesh2D, face_velocity, cell_velocity,
                               ::Type{T}) where {T<:AbstractFloat}
    nf = length(mesh.faces.c1)
    vx = zeros(T, nf)
    vy = zeros(T, nf)
    if face_velocity !== nothing
        if face_velocity isa AbstractMatrix
            size(face_velocity) == (nf, 2) || error("face_velocity must be nf x 2")
            vx .= T.(@view face_velocity[:, 1])
            vy .= T.(@view face_velocity[:, 2])
        else
            length(face_velocity) == nf || error("face_velocity length must match face count")
            for f in 1:nf
                v = face_velocity[f]
                vx[f] = T(v[1])
                vy[f] = T(v[2])
            end
        end
    elseif cell_velocity !== nothing
        n = length(mesh.cells)
        cm = _velocity_matrix(cell_velocity, n)
        for f in 1:nf
            i = mesh.faces.c1[f]
            j = mesh.faces.c2[f]
            if j > 0
                vx[f] = T(0.5 * (cm[i, 1] + cm[j, 1]))
                vy[f] = T(0.5 * (cm[i, 2] + cm[j, 2]))
            else
                vx[f] = T(cm[i, 1])
                vy[f] = T(cm[i, 2])
            end
        end
    end
    return vx, vy
end

function arepo_mesh_arrays(mesh::PolygonMesh2D; T::Type{<:AbstractFloat} = Float64,
                           index_type::Type{<:Integer} = Int32,
                           face_velocity = nothing, cell_velocity = nothing)
    f = mesh.faces
    offsets, faces, signs = _cell_face_csr(mesh, index_type)
    fvx, fvy = _face_velocity_arrays(mesh, face_velocity, cell_velocity, T)
    return ArepoMeshArrays2D(
        index_type.(f.c1),
        index_type.(f.c2),
        offsets,
        faces,
        signs,
        T.(cell_areas(mesh)),
        T.(f.area),
        T.(f.normal[:, 1]),
        T.(f.normal[:, 2]),
        fvx,
        fvy,
    )
end

function _backend_zeros(be, ::Type{T}, n::Integer) where {T}
    try
        return KA.zeros(be, T, n; unified = false)
    catch err
        err isa MethodError || rethrow()
        return KA.zeros(be, T, n)
    end
end

function _backend_copy(be, a::AbstractVector, ::Type{T}) where {T}
    out = _backend_zeros(be, T, length(a))
    copyto!(out, T.(a))
    return out
end

"""
    to_backend(be, mesh_arrays; T=Float32, index_type=Int32)
    to_backend(be, state; T=Float32)

Copy PowerFoam mesh/state arrays to a KernelAbstractions backend.  `be` can be
`KernelAbstractions.CPU()` or a GPU backend registered by a package such as Metal.
"""
function to_backend(be, mesh::ArepoMeshArrays2D; T::Type{<:AbstractFloat} = Float32,
                    index_type::Type{<:Integer} = Int32)
    return ArepoMeshArrays2D(
        _backend_copy(be, mesh.c1, index_type),
        _backend_copy(be, mesh.c2, index_type),
        _backend_copy(be, mesh.cell_face_offsets, index_type),
        _backend_copy(be, mesh.cell_faces, index_type),
        _backend_copy(be, mesh.cell_face_signs, index_type),
        _backend_copy(be, mesh.volume, T),
        _backend_copy(be, mesh.face_area, T),
        _backend_copy(be, mesh.normal_x, T),
        _backend_copy(be, mesh.normal_y, T),
        _backend_copy(be, mesh.face_vx, T),
        _backend_copy(be, mesh.face_vy, T),
    )
end

function to_backend(be, state::EulerState2D; T::Type{<:AbstractFloat} = Float32)
    return EulerState2D(_backend_copy(be, state.D, T),
                        _backend_copy(be, state.Mx, T),
                        _backend_copy(be, state.My, T),
                        _backend_copy(be, state.E, T))
end

function _expand_cell_value(x, n::Integer, ::Type{T}) where {T}
    if x isa Number
        return fill(T(x), n)
    end
    length(x) == n || error("cell field length $(length(x)) does not match cell count $n")
    return T.(collect(x))
end

"""
    euler_state_2d(mesh; rho, vx, vy, pressure, gamma=5/3, T=Float64)

Create conserved cell averages `(rho, rho*vx, rho*vy, rho*etot)` on a
`PolygonMesh2D`.  Scalar fields are broadcast to all cells.
"""
function euler_state_2d(mesh::PolygonMesh2D; rho = 1.0, vx = 0.0, vy = 0.0,
                        pressure = 1.0, gamma::Real = 5/3,
                        T::Type{<:AbstractFloat} = Float64)
    n = length(mesh.cells)
    r = _expand_cell_value(rho, n, T)
    ux = _expand_cell_value(vx, n, T)
    uy = _expand_cell_value(vy, n, T)
    p = _expand_cell_value(pressure, n, T)
    D = copy(r)
    Mx = r .* ux
    My = r .* uy
    E = p ./ T(gamma - 1) .+ T(0.5) .* r .* (ux .* ux .+ uy .* uy)
    return EulerState2D(D, Mx, My, E)
end

function primitive_to_conserved_2d!(state::EulerState2D, rho, vx, vy, pressure;
                                    gamma::Real)
    n = length(state.D)
    length(rho) == n && length(vx) == n && length(vy) == n && length(pressure) == n ||
        error("primitive arrays must all match state length")
    T = eltype(state.D)
    @inbounds for i in 1:n
        r = T(rho[i])
        ux = T(vx[i])
        uy = T(vy[i])
        p = T(pressure[i])
        state.D[i] = r
        state.Mx[i] = r * ux
        state.My[i] = r * uy
        state.E[i] = p / T(gamma - 1) + T(0.5) * r * (ux * ux + uy * uy)
    end
    return state
end

function conserved_to_primitive_2d(state::EulerState2D; gamma::Real)
    D = Array(state.D)
    Mx = Array(state.Mx)
    My = Array(state.My)
    E = Array(state.E)
    vx = Mx ./ D
    vy = My ./ D
    pressure = (gamma - 1) .* (E .- 0.5 .* (Mx .* Mx .+ My .* My) ./ D)
    return (; rho = D, vx, vy, pressure)
end

primitive_work_2d(state::EulerState2D) =
    PrimitiveState2D((similar(state.D) for _ in 1:4)...)

@kernel function _conserved_to_primitive_2d_k!(rho, vx, vy, pressure,
                                               @Const(D), @Const(Mx),
                                               @Const(My), @Const(E), gamma)
    i = @index(Global, Linear)
    T = eltype(rho)
    @inbounds begin
        r = D[i]
        ux = Mx[i] / r
        uy = My[i] / r
        rho[i] = r
        vx[i] = ux
        vy[i] = uy
        pressure[i] = (gamma - one(T)) *
                      (E[i] - T(0.5) * (Mx[i] * ux + My[i] * uy))
    end
end

function conserved_to_primitive_2d!(out::PrimitiveState2D, state::EulerState2D;
                                    gamma::Real, synchronize::Bool = true)
    n = length(state.D)
    length(out.rho) == n && length(out.vx) == n && length(out.vy) == n &&
        length(out.pressure) == n || error("primitive work arrays must match state length")
    be = KA.get_backend(state.D)
    T = eltype(state.D)
    _conserved_to_primitive_2d_k!(be)(out.rho, out.vx, out.vy, out.pressure,
                                      state.D, state.Mx, state.My, state.E,
                                      T(gamma); ndrange = n)
    synchronize && KA.synchronize(be)
    return out
end

primitive_to_arrays_2d(prim::PrimitiveState2D) =
    (; rho = Array(prim.rho), vx = Array(prim.vx), vy = Array(prim.vy),
     pressure = Array(prim.pressure))

hydro_work_2d(state::EulerState2D, mesh::ArepoMeshArrays2D) =
    FaceFluxWork2D(similar(state.D, length(mesh.c1)),
                   similar(state.D, length(mesh.c1)),
                   similar(state.D, length(mesh.c1)),
                   similar(state.D, length(mesh.c1)))

hydro_gradient_work_2d(rho::AbstractVector) =
    HydroGradients2D((similar(rho) for _ in 1:8)...)

face_prediction_work_2d(mesh::ArepoMeshArrays2D) =
    FaceStates2D(similar(mesh.face_area, 4 * length(mesh.face_area)),
                 similar(mesh.face_area, 4 * length(mesh.face_area)))

function face_states_to_arrays(states::FaceStates2D)
    nf = length(states.left) ÷ 4
    left = reshape(Array(states.left), nf, 4)
    right = reshape(Array(states.right), nf, 4)
    return (; left = (; rho = left[:, 1], vx = left[:, 2], vy = left[:, 3],
                      pressure = left[:, 4]),
            right = (; rho = right[:, 1], vx = right[:, 2], vy = right[:, 3],
                       pressure = right[:, 4]))
end

function hydro_gradients_to_arrays(g::HydroGradients2D)
    drho = hcat(Array(g.drho_x), Array(g.drho_y))
    dpress = hcat(Array(g.dpress_x), Array(g.dpress_y))
    n = length(g.drho_x)
    dvel = Array{eltype(drho)}(undef, n, 2, 2)
    dvel[:, 1, 1] .= Array(g.dvelx_x); dvel[:, 1, 2] .= Array(g.dvelx_y)
    dvel[:, 2, 1] .= Array(g.dvely_x); dvel[:, 2, 2] .= Array(g.dvely_y)
    return (; drho, dvel, dpress)
end

@inline function _pressure_2d(D::T, Mx::T, My::T, E::T, gm1::T, small::T) where {T}
    kinetic = T(0.5) * (Mx * Mx + My * My) / D
    return max(gm1 * (E - kinetic), small)
end

@inline function _normal_flux_2d(D::T, Mx::T, My::T, E::T, nx::T, ny::T,
                                 wx::T, wy::T, gamma::T, gm1::T, small::T) where {T}
    vx = Mx / D
    vy = My / D
    p = _pressure_2d(D, Mx, My, E, gm1, small)
    un = vx * nx + vy * ny
    wn = wx * nx + wy * ny
    urel = un - wn
    return (D * urel,
            Mx * urel + p * nx,
            My * urel + p * ny,
            E * urel + p * un,
            urel,
            sqrt(gamma * p / D))
end

@inline function _hll_or_llf_flux_2d(Dl::T, Mxl::T, Myl::T, El::T,
                                     Dr::T, Mxr::T, Myr::T, Er::T,
                                     nx::T, ny::T, wx::T, wy::T,
                                     gamma::T, solver::Int, small::T) where {T}
    gm1 = gamma - one(T)
    FL = _normal_flux_2d(Dl, Mxl, Myl, El, nx, ny, wx, wy, gamma, gm1, small)
    FR = _normal_flux_2d(Dr, Mxr, Myr, Er, nx, ny, wx, wy, gamma, gm1, small)
    if solver == 1
        a = max(abs(FL[5]) + FL[6], abs(FR[5]) + FR[6])
        h = T(0.5)
        return (h * (FL[1] + FR[1] - a * (Dr  - Dl)),
                h * (FL[2] + FR[2] - a * (Mxr - Mxl)),
                h * (FL[3] + FR[3] - a * (Myr - Myl)),
                h * (FL[4] + FR[4] - a * (Er  - El)))
    end
    sl = min(FL[5] - FL[6], FR[5] - FR[6])
    sr = max(FL[5] + FL[6], FR[5] + FR[6])
    sl >= zero(T) && return (FL[1], FL[2], FL[3], FL[4])
    sr <= zero(T) && return (FR[1], FR[2], FR[3], FR[4])
    denom = sr - sl
    return ((sr * FL[1] - sl * FR[1] + sl * sr * (Dr  - Dl)) / denom,
            (sr * FL[2] - sl * FR[2] + sl * sr * (Mxr - Mxl)) / denom,
            (sr * FL[3] - sl * FR[3] + sl * sr * (Myr - Myl)) / denom,
            (sr * FL[4] - sl * FR[4] + sl * sr * (Er  - El)) / denom)
end

@kernel function _face_flux_2d_k!(FD, FMx, FMy, FE,
                                  @Const(D), @Const(Mx), @Const(My), @Const(E),
                                  @Const(c1), @Const(c2), @Const(area),
                                  @Const(nx), @Const(ny), @Const(wx), @Const(wy),
                                  gamma, solver::Int, small)
    f = @index(Global, Linear)
    T = eltype(FD)
    @inbounds begin
        i = Int(c1[f])
        j = Int(c2[f])
        if j <= 0
            FD[f] = zero(T); FMx[f] = zero(T); FMy[f] = zero(T); FE[f] = zero(T)
        else
            flux = _hll_or_llf_flux_2d(D[i], Mx[i], My[i], E[i],
                                       D[j], Mx[j], My[j], E[j],
                                       nx[f], ny[f], wx[f], wy[f],
                                       gamma, solver, small)
            a = area[f]
            FD[f] = flux[1] * a
            FMx[f] = flux[2] * a
            FMy[f] = flux[3] * a
            FE[f] = flux[4] * a
        end
    end
end

@kernel function _cell_update_2d_k!(D, Mx, My, E,
                                    @Const(FD), @Const(FMx), @Const(FMy), @Const(FE),
                                    @Const(old_volume), @Const(new_volume),
                                    @Const(offsets), @Const(faces),
                                    @Const(signs), dt)
    i = @index(Global, Linear)
    T = eltype(D)
    dD = zero(T); dMx = zero(T); dMy = zero(T); dE = zero(T)
    @inbounds begin
        for p in Int(offsets[i]):(Int(offsets[i + 1]) - 1)
            f = Int(faces[p])
            s = T(signs[p])
            dD += s * FD[f]
            dMx += s * FMx[f]
            dMy += s * FMy[f]
            dE += s * FE[f]
        end
        vold = old_volume[i]
        vnew = new_volume[i]
        D[i] = (D[i] * vold + dt * dD) / vnew
        Mx[i] = (Mx[i] * vold + dt * dMx) / vnew
        My[i] = (My[i] * vold + dt * dMy) / vnew
        E[i] = (E[i] * vold + dt * dE) / vnew
    end
end

function _solver_code(riemann::Symbol)
    riemann === :hll && return 0
    (riemann === :llf || riemann === :rusanov) && return 1
    error("finite_volume_step_2d!: unsupported riemann=$riemann; use :hll or :llf")
end

"""
    finite_volume_step_2d!(state, mesh; dt, gamma, riemann=:hll, work=hydro_work_2d(...))

Advance one first-order Euler step on an AREPO-shaped 2-D face table.  Boundary
faces (`c2 == 0`) are closed by setting their flux to zero.  Internal face fluxes
are stored once with normals pointing from `c1` to `c2`; the cell update gathers
those fluxes conservatively through the CSR table.
"""
function finite_volume_step_2d!(state::EulerState2D, mesh::ArepoMeshArrays2D;
                                dt::Real, gamma::Real, riemann::Symbol = :hll,
                                work::Union{Nothing,FaceFluxWork2D} = nothing,
                                new_volume = mesh.volume,
                                small_pressure::Real = 1e-12)
    be = KA.get_backend(state.D)
    T = eltype(state.D)
    w = work === nothing ? hydro_work_2d(state, mesh) : work
    _face_flux_2d_k!(be)(w.FD, w.FMx, w.FMy, w.FE,
                         state.D, state.Mx, state.My, state.E,
                         mesh.c1, mesh.c2, mesh.face_area, mesh.normal_x, mesh.normal_y,
                         mesh.face_vx, mesh.face_vy,
                         T(gamma), _solver_code(riemann), T(small_pressure);
                         ndrange = length(mesh.c1))
    _cell_update_2d_k!(be)(state.D, state.Mx, state.My, state.E,
                           w.FD, w.FMx, w.FMy, w.FE,
                           mesh.volume, new_volume, mesh.cell_face_offsets, mesh.cell_faces,
                           mesh.cell_face_signs, T(dt);
                           ndrange = length(state.D))
    KA.synchronize(be)
    return state
end

@inline function _prim_to_cons2(rho::T, vx::T, vy::T, p::T,
                                gamma::T) where {T}
    return (rho, rho * vx, rho * vy,
            p / (gamma - one(T)) + T(0.5) * rho * (vx * vx + vy * vy))
end

@kernel function _face_flux_from_predicted_2d_k!(FD, FMx, FMy, FE,
                                                 @Const(left), @Const(right),
                                                 @Const(c2), @Const(area),
                                                 @Const(nx), @Const(ny),
                                                 @Const(wx), @Const(wy),
                                                 gamma, solver::Int, small)
    f = @index(Global, Linear)
    T = eltype(FD)
    nface = Int(length(FD))
    @inbounds begin
        if Int(c2[f]) <= 0
            FD[f] = zero(T); FMx[f] = zero(T); FMy[f] = zero(T); FE[f] = zero(T)
        else
            Dl = max(left[f], small)
            ulx = left[nface + f] + wx[f]
            uly = left[2 * nface + f] + wy[f]
            pl = max(left[3 * nface + f], small)
            Dr = max(right[f], small)
            urx = right[nface + f] + wx[f]
            ury = right[2 * nface + f] + wy[f]
            pr = max(right[3 * nface + f], small)
            CL = _prim_to_cons2(Dl, ulx, uly, pl, gamma)
            CR = _prim_to_cons2(Dr, urx, ury, pr, gamma)
            flux = _hll_or_llf_flux_2d(CL[1], CL[2], CL[3], CL[4],
                                       CR[1], CR[2], CR[3], CR[4],
                                       nx[f], ny[f], wx[f], wy[f],
                                       gamma, solver, small)
            a = area[f]
            FD[f] = flux[1] * a
            FMx[f] = flux[2] * a
            FMy[f] = flux[3] * a
            FE[f] = flux[4] * a
        end
    end
end

@inline function _periodic_delta2(d::T, box::T) where {T}
    box <= zero(T) && return d
    half = T(0.5) * box
    d < -half && return d + box
    d > half && return d - box
    return d
end

@inline function _solve_lsq2(x11::T, x12::T, x22::T, y1::T, y2::T) where {T}
    det = x11 * x22 - x12 * x12
    scale = max(abs(x11 * x22), one(T))
    abs(det) <= eps(T) * scale && return (zero(T), zero(T))
    invdet = one(T) / det
    return ((x22 * y1 - x12 * y2) * invdet,
            (x11 * y2 - x12 * y1) * invdet)
end

@inline function _limit_gradient2(dx::T, dy::T, phi::T, minphi::T, maxphi::T,
                                  gx::T, gy::T) where {T}
    dp = gx * dx + gy * dy
    if dp > zero(T)
        if phi + dp > maxphi
            fac = maxphi > phi ? (maxphi - phi) / dp : zero(T)
            gx *= fac; gy *= fac
        end
    elseif dp < zero(T)
        if phi + dp < minphi
            fac = minphi < phi ? (minphi - phi) / dp : zero(T)
            gx *= fac; gy *= fac
        end
    end
    return (gx, gy)
end

@inline function _limit_vel_gradient2(dx::T, dy::T, csnd::T, gx::T, gy::T) where {T}
    dv = abs(gx * dx + gy * dy)
    if dv > csnd
        fac = csnd / dv
        gx *= fac; gy *= fac
    end
    return (gx, gy)
end

@kernel function _pack_gradient_primitive_cell_data_2d_k!(
    cell_data, @Const(rho), @Const(velx), @Const(vely),
    @Const(pressure), @Const(center_x), @Const(center_y), gamma)
    i = @index(Global, Linear)
    n = Int(length(rho))
    @inbounds begin
        cell_data[i] = rho[i]
        cell_data[n + i] = velx[i]
        cell_data[2 * n + i] = vely[i]
        cell_data[3 * n + i] = pressure[i]
        cell_data[4 * n + i] = sqrt(gamma * pressure[i] / rho[i])
        cell_data[5 * n + i] = center_x[i]
        cell_data[6 * n + i] = center_y[i]
    end
end

@kernel function _pack_gradient_mesh_face_data_2d_k!(
    face_data, @Const(face_area), @Const(face_center_x), @Const(face_center_y))
    f = @index(Global, Linear)
    nf = Int(length(face_area))
    @inbounds begin
        face_data[f] = face_area[f]
        face_data[nf + f] = face_center_x[f]
        face_data[2 * nf + f] = face_center_y[f]
    end
end

@kernel function _unpack_gradients_2d_k!(
    drho_x, drho_y, dvelx_x, dvelx_y, dvely_x, dvely_y,
    dpress_x, dpress_y, @Const(grad_data))
    i = @index(Global, Linear)
    n = Int(length(drho_x))
    @inbounds begin
        drho_x[i] = grad_data[i]
        drho_y[i] = grad_data[n + i]
        dvelx_x[i] = grad_data[2 * n + i]
        dvelx_y[i] = grad_data[3 * n + i]
        dvely_x[i] = grad_data[4 * n + i]
        dvely_y[i] = grad_data[5 * n + i]
        dpress_x[i] = grad_data[6 * n + i]
        dpress_y[i] = grad_data[7 * n + i]
    end
end

@kernel function _pack_hydro_gradients_2d_k!(
    grad_data, @Const(drho_x), @Const(drho_y),
    @Const(dvelx_x), @Const(dvelx_y),
    @Const(dvely_x), @Const(dvely_y),
    @Const(dpress_x), @Const(dpress_y))
    i = @index(Global, Linear)
    n = Int(length(drho_x))
    @inbounds begin
        grad_data[i] = drho_x[i]
        grad_data[n + i] = drho_y[i]
        grad_data[2 * n + i] = dvelx_x[i]
        grad_data[3 * n + i] = dvelx_y[i]
        grad_data[4 * n + i] = dvely_x[i]
        grad_data[5 * n + i] = dvely_y[i]
        grad_data[6 * n + i] = dpress_x[i]
        grad_data[7 * n + i] = dpress_y[i]
    end
end

function _pack_hydro_gradients_backend(be, ::Type{T},
                                       gradients::HydroGradients2D) where {T}
    n = length(gradients.drho_x)
    grad_data = _backend_zeros(be, T, 8 * n)
    _pack_hydro_gradients_2d_k!(be)(
        grad_data, gradients.drho_x, gradients.drho_y,
        gradients.dvelx_x, gradients.dvelx_y,
        gradients.dvely_x, gradients.dvely_y,
        gradients.dpress_x, gradients.dpress_y;
        ndrange = n)
    return grad_data
end

@kernel function _gradients_from_mesh_2d_k!(
    grad_data, @Const(offsets), @Const(cell_faces), @Const(cell_signs),
    @Const(c1), @Const(c2), @Const(cell_data), @Const(face_data), box_size)
    i = @index(Global, Linear)
    T = eltype(cell_data)
    ncell = Int(length(offsets) - 1)
    nface = Int(length(c1))
    x11 = zero(T); x12 = zero(T); x22 = zero(T)
    yr1 = zero(T); yr2 = zero(T)
    yvx1 = zero(T); yvx2 = zero(T)
    yvy1 = zero(T); yvy2 = zero(T)
    yp1 = zero(T); yp2 = zero(T)
    minr = typemax(T); maxr = -typemax(T)
    minvx = typemax(T); maxvx = -typemax(T)
    minvy = typemax(T); maxvy = -typemax(T)
    minp = typemax(T); maxp = -typemax(T)
    r0 = cell_data[i]
    vx0 = cell_data[ncell + i]
    vy0 = cell_data[2 * ncell + i]
    p0 = cell_data[3 * ncell + i]
    csnd = cell_data[4 * ncell + i]
    cx = cell_data[5 * ncell + i]
    cy = cell_data[6 * ncell + i]
    @inbounds begin
        for p in Int(offsets[i]):(Int(offsets[i + 1]) - 1)
            f = Int(cell_faces[p])
            w = face_data[f]
            w > zero(T) || continue
            sign = Int(cell_signs[p])
            other = sign < 0 ? Int(c2[f]) : Int(c1[f])
            other > 0 || continue
            dx = _periodic_delta2(cell_data[5 * ncell + other] - cx, box_size)
            dy = _periodic_delta2(cell_data[6 * ncell + other] - cy, box_size)
            dist = sqrt(dx * dx + dy * dy)
            dist > zero(T) || continue
            invd = one(T) / dist
            nx = dx * invd; ny = dy * invd
            x11 += w * nx * nx
            x12 += w * nx * ny
            x22 += w * ny * ny

            ro = cell_data[other]
            vxo = cell_data[ncell + other]
            vyo = cell_data[2 * ncell + other]
            po = cell_data[3 * ncell + other]

            fac = w * (ro - r0) * invd
            yr1 += fac * nx; yr2 += fac * ny
            fac = w * (vxo - vx0) * invd
            yvx1 += fac * nx; yvx2 += fac * ny
            fac = w * (vyo - vy0) * invd
            yvy1 += fac * nx; yvy2 += fac * ny
            fac = w * (po - p0) * invd
            yp1 += fac * nx; yp2 += fac * ny

            minr = min(minr, ro); maxr = max(maxr, ro)
            minvx = min(minvx, vxo); maxvx = max(maxvx, vxo)
            minvy = min(minvy, vyo); maxvy = max(maxvy, vyo)
            minp = min(minp, po); maxp = max(maxp, po)
        end

        gr = _solve_lsq2(x11, x12, x22, yr1, yr2)
        gvx = _solve_lsq2(x11, x12, x22, yvx1, yvx2)
        gvy = _solve_lsq2(x11, x12, x22, yvy1, yvy2)
        gp = _solve_lsq2(x11, x12, x22, yp1, yp2)

        grx = gr[1]; gry = gr[2]
        gvxx = gvx[1]; gvxy = gvx[2]
        gvyx = gvy[1]; gvyy = gvy[2]
        gpx = gp[1]; gpy = gp[2]

        for p in Int(offsets[i]):(Int(offsets[i + 1]) - 1)
            f = Int(cell_faces[p])
            face_data[f] > zero(T) || continue
            dx = _periodic_delta2(face_data[nface + f] - cx, box_size)
            dy = _periodic_delta2(face_data[2 * nface + f] - cy, box_size)
            gr = _limit_gradient2(dx, dy, r0, minr, maxr, grx, gry)
            grx = gr[1]; gry = gr[2]
            gvx = _limit_gradient2(dx, dy, vx0, minvx, maxvx, gvxx, gvxy)
            gvxx = gvx[1]; gvxy = gvx[2]
            gvy = _limit_gradient2(dx, dy, vy0, minvy, maxvy, gvyx, gvyy)
            gvyx = gvy[1]; gvyy = gvy[2]
            gp = _limit_gradient2(dx, dy, p0, minp, maxp, gpx, gpy)
            gpx = gp[1]; gpy = gp[2]
        end

        for p in Int(offsets[i]):(Int(offsets[i + 1]) - 1)
            f = Int(cell_faces[p])
            face_data[f] > zero(T) || continue
            dx = _periodic_delta2(face_data[nface + f] - cx, box_size)
            dy = _periodic_delta2(face_data[2 * nface + f] - cy, box_size)
            gvx = _limit_vel_gradient2(dx, dy, csnd, gvxx, gvxy)
            gvxx = gvx[1]; gvxy = gvx[2]
            gvy = _limit_vel_gradient2(dx, dy, csnd, gvyx, gvyy)
            gvyx = gvy[1]; gvyy = gvy[2]
        end

        grad_data[i] = grx
        grad_data[ncell + i] = gry
        grad_data[2 * ncell + i] = gvxx
        grad_data[3 * ncell + i] = gvxy
        grad_data[4 * ncell + i] = gvyx
        grad_data[5 * ncell + i] = gvyy
        grad_data[6 * ncell + i] = gpx
        grad_data[7 * ncell + i] = gpy
    end
end

function calculate_gradients_from_mesh_2d!(out::HydroGradients2D,
                                           mesh::ArepoMeshArrays2D,
                                           prim::PrimitiveState2D,
                                           center, face_center;
                                           box_size::Real = 1.0,
                                           gamma::Real = 5/3,
                                           synchronize::Bool = true)
    n = length(prim.rho)
    nf = length(mesh.c1)
    size(center) == (n, 2) || error("center must be n x 2")
    size(face_center) == (nf, 2) || error("face_center must be nf x 2")
    be = KA.get_backend(out.drho_x)
    T = eltype(out.drho_x)
    cx = _backend_copy(be, collect(view(center, :, 1)), T)
    cy = _backend_copy(be, collect(view(center, :, 2)), T)
    fcx = _backend_copy(be, collect(view(face_center, :, 1)), T)
    fcy = _backend_copy(be, collect(view(face_center, :, 2)), T)
    calculate_gradients_from_mesh_2d!(out, mesh, prim, cx, cy, fcx, fcy;
                                      box_size, gamma, synchronize)
end

function calculate_gradients_from_mesh_2d!(out::HydroGradients2D,
                                           mesh::ArepoMeshArrays2D,
                                           prim::PrimitiveState2D,
                                           center_x, center_y,
                                           face_center_x, face_center_y;
                                           box_size::Real = 1.0,
                                           gamma::Real = 5/3,
                                           synchronize::Bool = true)
    n = length(prim.rho)
    nf = length(mesh.c1)
    length(center_x) == n || error("center_x has wrong length")
    length(center_y) == n || error("center_y has wrong length")
    length(face_center_x) == nf || error("face_center_x has wrong length")
    length(face_center_y) == nf || error("face_center_y has wrong length")
    be = KA.get_backend(out.drho_x)
    T = eltype(out.drho_x)
    cell_data = _backend_zeros(be, T, 7 * n)
    _pack_gradient_primitive_cell_data_2d_k!(be)(
        cell_data, prim.rho, prim.vx, prim.vy, prim.pressure,
        center_x, center_y, T(gamma);
        ndrange = n)
    face_data = _backend_zeros(be, T, 3 * nf)
    _pack_gradient_mesh_face_data_2d_k!(be)(
        face_data, mesh.face_area, face_center_x, face_center_y;
        ndrange = nf)
    grad_data = _backend_zeros(be, T, 8 * n)
    _gradients_from_mesh_2d_k!(be)(
        grad_data, mesh.cell_face_offsets, mesh.cell_faces,
        mesh.cell_face_signs, mesh.c1, mesh.c2, cell_data, face_data,
        T(box_size);
        ndrange = n)
    _unpack_gradients_2d_k!(be)(
        out.drho_x, out.drho_y, out.dvelx_x, out.dvelx_y,
        out.dvely_x, out.dvely_y, out.dpress_x, out.dpress_y,
        grad_data;
        ndrange = n)
    synchronize && KA.synchronize(be)
    return out
end

@kernel function _pack_predict_primitive_cell_data_2d_k!(
    cell_data, @Const(rho), @Const(velx), @Const(vely),
    @Const(pressure), @Const(center_x), @Const(center_y), @Const(dt))
    i = @index(Global, Linear)
    n = Int(length(rho))
    @inbounds begin
        cell_data[i] = rho[i]
        cell_data[n + i] = velx[i]
        cell_data[2 * n + i] = vely[i]
        cell_data[3 * n + i] = pressure[i]
        cell_data[4 * n + i] = center_x[i]
        cell_data[5 * n + i] = center_y[i]
        cell_data[6 * n + i] = dt[i]
    end
end

@kernel function _pack_predict_face_data_2d_k!(
    face_data, @Const(face_center_x), @Const(face_center_y),
    @Const(face_vx), @Const(face_vy), @Const(normal_x), @Const(normal_y))
    f = @index(Global, Linear)
    nf = Int(length(face_center_x))
    @inbounds begin
        face_data[f] = face_center_x[f]
        face_data[nf + f] = face_center_y[f]
        face_data[2 * nf + f] = face_vx[f]
        face_data[3 * nf + f] = face_vy[f]
        face_data[4 * nf + f] = normal_x[f]
        face_data[5 * nf + f] = normal_y[f]
    end
end

@inline function _predict_primitive_face_state2(cell::Int, f::Int, ncell::Int,
                                                nface::Int, cell_data,
                                                grad_data, face_data,
                                                box_size::T, gamma::T) where {T}
    r0 = cell_data[cell]
    vx0 = cell_data[ncell + cell]
    vy0 = cell_data[2 * ncell + cell]
    p0 = cell_data[3 * ncell + cell]
    cx = cell_data[4 * ncell + cell]
    cy = cell_data[5 * ncell + cell]
    dt = cell_data[6 * ncell + cell]

    grx = grad_data[cell]
    gry = grad_data[ncell + cell]
    gvxx = grad_data[2 * ncell + cell]
    gvxy = grad_data[3 * ncell + cell]
    gvyx = grad_data[4 * ncell + cell]
    gvyy = grad_data[5 * ncell + cell]
    gpx = grad_data[6 * ncell + cell]
    gpy = grad_data[7 * ncell + cell]

    fx = face_data[f]
    fy = face_data[nface + f]
    wx = face_data[2 * nface + f]
    wy = face_data[3 * nface + f]

    dx = _periodic_delta2(fx - cx, box_size)
    dy = _periodic_delta2(fy - cy, box_size)

    vx = vx0 - wx
    vy = vy0 - wy

    dr_time = -dt * (vx * grx + r0 * gvxx + vy * gry + r0 * gvyy)
    dvx_time = -dt * (gpx / r0 + vx * gvxx + vy * gvxy)
    dvy_time = -dt * (gpy / r0 + vx * gvyx + vy * gvyy)
    dp_time = -dt * (gamma * p0 * (gvxx + gvyy) + vx * gpx + vy * gpy)

    dr_space = grx * dx + gry * dy
    dvx_space = gvxx * dx + gvxy * dy
    dvy_space = gvyx * dx + gvyy * dy
    dp_space = gpx * dx + gpy * dy

    r = r0
    p = p0
    if r0 > zero(T) && r0 + dr_time + dr_space >= zero(T) &&
       p0 + dp_time + dp_space >= zero(T)
        r += dr_time + dr_space
        vx += dvx_time + dvx_space
        vy += dvy_time + dvy_space
        p += dp_time + dp_space
    end
    return (r, vx, vy, p)
end

@kernel function _predict_face_states_2d_k!(
    left, right, @Const(c1), @Const(c2),
    @Const(cell_data), @Const(grad_data), @Const(face_data),
    box_size, gamma)
    f = @index(Global, Linear)
    ncell = Int(length(cell_data) ÷ 7)
    nface = Int(length(left) ÷ 4)
    @inbounds begin
        i = Int(c1[f])
        j = Int(c2[f])
        L = _predict_primitive_face_state2(i, f, ncell, nface, cell_data,
                                           grad_data, face_data, box_size, gamma)
        left[f] = L[1]
        left[nface + f] = L[2]
        left[2 * nface + f] = L[3]
        left[3 * nface + f] = L[4]
        if j > 0
            R = _predict_primitive_face_state2(j, f, ncell, nface, cell_data,
                                               grad_data, face_data, box_size, gamma)
            right[f] = R[1]
            right[nface + f] = R[2]
            right[2 * nface + f] = R[3]
            right[3 * nface + f] = R[4]
        else
            right[f] = L[1]
            right[nface + f] = L[2]
            right[2 * nface + f] = L[3]
            right[3 * nface + f] = L[4]
        end
    end
end

function predict_face_states_2d!(states::FaceStates2D, mesh::ArepoMeshArrays2D,
                                 gradients::HydroGradients2D,
                                 prim::PrimitiveState2D, center, face_center;
                                 dt_extrapolation = nothing,
                                 box_size::Real = 1.0,
                                 gamma::Real = 5/3,
                                 synchronize::Bool = true)
    n = length(prim.rho)
    nf = length(mesh.c1)
    size(center) == (n, 2) || error("center must be n x 2")
    size(face_center) == (nf, 2) || error("face_center must be nf x 2")
    be = KA.get_backend(states.left)
    T = eltype(states.left)
    dt = dt_extrapolation === nothing ?
         _backend_copy(be, fill(zero(T), n), T) :
         _backend_copy(be, Array(dt_extrapolation), T)
    cx = _backend_copy(be, collect(view(center, :, 1)), T)
    cy = _backend_copy(be, collect(view(center, :, 2)), T)
    fcx = _backend_copy(be, collect(view(face_center, :, 1)), T)
    fcy = _backend_copy(be, collect(view(face_center, :, 2)), T)
    predict_face_states_2d!(states, mesh, gradients, prim, cx, cy, fcx, fcy;
                            dt_extrapolation = dt, box_size, gamma, synchronize)
end

function predict_face_states_2d!(states::FaceStates2D, mesh::ArepoMeshArrays2D,
                                 gradients::HydroGradients2D,
                                 prim::PrimitiveState2D,
                                 center_x, center_y,
                                 face_center_x, face_center_y;
                                 dt_extrapolation = nothing,
                                 box_size::Real = 1.0,
                                 gamma::Real = 5/3,
                                 synchronize::Bool = true)
    n = length(prim.rho)
    nf = length(mesh.c1)
    length(center_x) == n || error("center_x has wrong length")
    length(center_y) == n || error("center_y has wrong length")
    length(face_center_x) == nf || error("face_center_x has wrong length")
    length(face_center_y) == nf || error("face_center_y has wrong length")
    length(states.left) == 4nf || error("left face-state buffer has wrong length")
    length(states.right) == 4nf || error("right face-state buffer has wrong length")
    be = KA.get_backend(states.left)
    T = eltype(states.left)
    dt = dt_extrapolation === nothing ?
         _backend_copy(be, fill(zero(T), n), T) :
         dt_extrapolation
    cell_data = _backend_zeros(be, T, 7 * n)
    _pack_predict_primitive_cell_data_2d_k!(be)(
        cell_data, prim.rho, prim.vx, prim.vy, prim.pressure,
        center_x, center_y, dt;
        ndrange = n)
    grad_data = _pack_hydro_gradients_backend(be, T, gradients)
    face_data = _backend_zeros(be, T, 6 * nf)
    _pack_predict_face_data_2d_k!(be)(
        face_data, face_center_x, face_center_y, mesh.face_vx, mesh.face_vy,
        mesh.normal_x, mesh.normal_y;
        ndrange = nf)
    _predict_face_states_2d_k!(be)(states.left, states.right, mesh.c1, mesh.c2,
                                   cell_data, grad_data, face_data,
                                   T(box_size), T(gamma);
                                   ndrange = nf)
    synchronize && KA.synchronize(be)
    return states
end

function finite_volume_reconstructed_step_2d!(
    state::EulerState2D, mesh::ArepoMeshArrays2D, gradients::HydroGradients2D,
    prim::PrimitiveState2D, center, face_center;
    dt::Real, gamma::Real, riemann::Symbol = :hll,
    dt_extrapolation = nothing,
    work::Union{Nothing,FaceFluxWork2D} = nothing,
    states::Union{Nothing,FaceStates2D} = nothing,
    new_volume = mesh.volume,
    box_size::Real = 1.0,
    small_pressure::Real = 1e-12,
    synchronize::Bool = true)
    n = length(prim.rho)
    nf = length(mesh.c1)
    size(center) == (n, 2) || error("center must be n x 2")
    size(face_center) == (nf, 2) || error("face_center must be nf x 2")
    be = KA.get_backend(state.D)
    T = eltype(state.D)
    half_dt = dt_extrapolation === nothing ? fill(T(0.5 * dt), n) :
              dt_extrapolation
    s = states === nothing ? face_prediction_work_2d(mesh) : states
    predict_face_states_2d!(s, mesh, gradients, prim, center, face_center;
                            dt_extrapolation = half_dt, box_size, gamma,
                            synchronize = false)
    w = work === nothing ? hydro_work_2d(state, mesh) : work
    _face_flux_from_predicted_2d_k!(be)(w.FD, w.FMx, w.FMy, w.FE,
                                        s.left, s.right, mesh.c2, mesh.face_area,
                                        mesh.normal_x, mesh.normal_y,
                                        mesh.face_vx, mesh.face_vy,
                                        T(gamma), _solver_code(riemann),
                                        T(small_pressure);
                                        ndrange = nf)
    _cell_update_2d_k!(be)(state.D, state.Mx, state.My, state.E,
                           w.FD, w.FMx, w.FMy, w.FE,
                           mesh.volume, new_volume, mesh.cell_face_offsets,
                           mesh.cell_faces, mesh.cell_face_signs, T(dt);
                           ndrange = length(state.D))
    synchronize && KA.synchronize(be)
    return state
end

function finite_volume_reconstructed_step_2d!(
    state::EulerState2D, mesh::ArepoMeshArrays2D, gradients::HydroGradients2D,
    prim::PrimitiveState2D, center_x, center_y, face_center_x, face_center_y;
    dt::Real, gamma::Real, riemann::Symbol = :hll,
    dt_extrapolation = nothing,
    work::Union{Nothing,FaceFluxWork2D} = nothing,
    states::Union{Nothing,FaceStates2D} = nothing,
    new_volume = mesh.volume,
    box_size::Real = 1.0,
    small_pressure::Real = 1e-12,
    synchronize::Bool = true)
    n = length(prim.rho)
    nf = length(mesh.c1)
    be = KA.get_backend(state.D)
    T = eltype(state.D)
    half_dt = dt_extrapolation === nothing ?
              _backend_copy(be, fill(T(0.5 * dt), n), T) :
              dt_extrapolation
    s = states === nothing ? face_prediction_work_2d(mesh) : states
    predict_face_states_2d!(s, mesh, gradients, prim, center_x, center_y,
                            face_center_x, face_center_y;
                            dt_extrapolation = half_dt, box_size, gamma,
                            synchronize = false)
    w = work === nothing ? hydro_work_2d(state, mesh) : work
    _face_flux_from_predicted_2d_k!(be)(w.FD, w.FMx, w.FMy, w.FE,
                                        s.left, s.right, mesh.c2, mesh.face_area,
                                        mesh.normal_x, mesh.normal_y,
                                        mesh.face_vx, mesh.face_vy,
                                        T(gamma), _solver_code(riemann),
                                        T(small_pressure);
                                        ndrange = nf)
    _cell_update_2d_k!(be)(state.D, state.Mx, state.My, state.E,
                           w.FD, w.FMx, w.FMy, w.FE,
                           mesh.volume, new_volume, mesh.cell_face_offsets,
                           mesh.cell_faces, mesh.cell_face_signs, T(dt);
                           ndrange = length(state.D))
    synchronize && KA.synchronize(be)
    return state
end

function _mesh_velocity_from_input(mesh::PolygonMesh2D, state::EulerState2D, mesh_velocity;
                                   gamma::Real)
    n = length(mesh.cells)
    if mesh_velocity === nothing
        p = conserved_to_primitive_2d(state; gamma)
        return hcat(p.vx, p.vy)
    elseif mesh_velocity isa AbstractMatrix
        size(mesh_velocity) == (n, 2) || error("mesh_velocity matrix must be n x 2")
        return Matrix{Float64}(mesh_velocity)
    else
        length(mesh_velocity) == n || error("mesh_velocity length must match cell count")
        out = Matrix{Float64}(undef, n, 2)
        for i in 1:n
            v = mesh_velocity[i]
            out[i, 1] = float(v[1])
            out[i, 2] = float(v[2])
        end
        return out
    end
end

function advect_generators_2d(points::AbstractMatrix, velocity::AbstractMatrix, dt::Real,
                              domain; boundary::Symbol = :clamp)
    size(points, 2) == 2 || error("points must be n x 2")
    size(velocity) == size(points) || error("velocity must have the same shape as points")
    out = Matrix{Float64}(points)
    @inbounds for i in axes(out, 1)
        out[i, 1] += dt * velocity[i, 1]
        out[i, 2] += dt * velocity[i, 2]
    end
    xmin, xmax = domain[1]
    ymin, ymax = domain[2]
    if boundary === :clamp
        @inbounds for i in axes(out, 1)
            out[i, 1] = clamp(out[i, 1], xmin, xmax)
            out[i, 2] = clamp(out[i, 2], ymin, ymax)
        end
    elseif boundary === :periodic
        lx = xmax - xmin
        ly = ymax - ymin
        @inbounds for i in axes(out, 1)
            out[i, 1] = xmin + mod(out[i, 1] - xmin, lx)
            out[i, 2] = ymin + mod(out[i, 2] - ymin, ly)
        end
    elseif boundary !== :none
        error("unknown moving mesh boundary=$boundary; use :clamp, :periodic, or :none")
    end
    return out
end

function _backend_float_type(a)
    T = eltype(a)
    T <: AbstractFloat || error("state arrays must have floating element type")
    return T
end

function _backend_index_type(a)
    T = eltype(a)
    T <: Integer || error("mesh index arrays must have integer element type")
    return T
end

"""
    moving_mesh_step_2d!(state, mesh; dt, gamma, mesh_velocity=nothing,
                         riemann=:hll, backend=nothing, boundary=:clamp)

Advance one ALE moving-mesh step.  `mesh` is the host `PolygonMesh2D`; `state`
may live on CPU arrays or a GPU backend.  If `mesh_velocity` is omitted, the mesh
generators move with the current fluid velocity.  The old mesh supplies the
moving-face ALE fluxes, generator points are advanced by `dt`, the Voronoi mesh is
rebuilt on the host, and the conserved cell integrals are divided by the new cell
volumes.

Returns `(mesh, geom, state, mesh_velocity)`, where `mesh` is the rebuilt host
mesh and `geom` is the backend-staged new mesh arrays.
"""
function moving_mesh_step_2d!(state::EulerState2D, mesh::PolygonMesh2D;
                              dt::Real, gamma::Real, mesh_velocity = nothing,
                              riemann::Symbol = :hll, backend = nothing,
                              boundary::Symbol = :clamp,
                              small_pressure::Real = 1e-12)
    vmesh = _mesh_velocity_from_input(mesh, state, mesh_velocity; gamma)
    old_host = arepo_mesh_arrays(mesh; T = Float64, cell_velocity = vmesh)
    new_points = advect_generators_2d(mesh.generators, vmesh, dt, mesh.domain; boundary)
    new_mesh = power_diagram(PowerSites2D(new_points; weights = mesh.weights, domain = mesh.domain))

    be = backend === nothing ? KA.get_backend(state.D) : backend
    T = _backend_float_type(state.D)
    I = _backend_index_type(old_host.c1)
    old_geom = to_backend(be, old_host; T, index_type = I)
    new_geom = to_backend(be, arepo_mesh_arrays(new_mesh; T = Float64); T, index_type = I)
    work = hydro_work_2d(state, old_geom)
    finite_volume_step_2d!(state, old_geom; dt, gamma, riemann, work,
                           new_volume = new_geom.volume, small_pressure)
    return (; mesh = new_mesh, geom = new_geom, state, mesh_velocity = vmesh)
end

function moving_mesh_reconstructed_step_2d!(state::EulerState2D, mesh::PolygonMesh2D;
                                            dt::Real, gamma::Real,
                                            mesh_velocity = nothing,
                                            riemann::Symbol = :hll,
                                            backend = nothing,
                                            boundary::Symbol = :clamp,
                                            box_size::Real = 0.0,
                                            small_pressure::Real = 1e-12)
    vmesh = _mesh_velocity_from_input(mesh, state, mesh_velocity; gamma)
    old_host = arepo_mesh_arrays(mesh; T = Float64, cell_velocity = vmesh)
    new_points = advect_generators_2d(mesh.generators, vmesh, dt, mesh.domain; boundary)
    new_mesh = power_diagram(PowerSites2D(new_points; weights = mesh.weights, domain = mesh.domain))

    be = backend === nothing ? KA.get_backend(state.D) : backend
    T = _backend_float_type(state.D)
    I = _backend_index_type(old_host.c1)
    old_geom = to_backend(be, old_host; T, index_type = I)
    new_geom = to_backend(be, arepo_mesh_arrays(new_mesh; T = Float64); T, index_type = I)

    center = cell_centroids(mesh)
    face_center = mesh.faces.center
    cx = _backend_copy(be, collect(view(center, :, 1)), T)
    cy = _backend_copy(be, collect(view(center, :, 2)), T)
    fcx = _backend_copy(be, collect(view(face_center, :, 1)), T)
    fcy = _backend_copy(be, collect(view(face_center, :, 2)), T)

    prim = primitive_work_2d(state)
    conserved_to_primitive_2d!(prim, state; gamma, synchronize = false)
    gradients = hydro_gradient_work_2d(prim.rho)
    calculate_gradients_from_mesh_2d!(gradients, old_geom, prim, cx, cy, fcx, fcy;
                                      box_size, gamma, synchronize = false)
    work = hydro_work_2d(state, old_geom)
    states = face_prediction_work_2d(old_geom)
    finite_volume_reconstructed_step_2d!(state, old_geom, gradients, prim,
                                         cx, cy, fcx, fcy;
                                         dt, gamma, riemann, work, states,
                                         new_volume = new_geom.volume,
                                         box_size, small_pressure)
    return (; mesh = new_mesh, geom = new_geom, state, mesh_velocity = vmesh)
end

function total_conserved_2d(state::EulerState2D, mesh::ArepoMeshArrays2D)
    D = Array(state.D); Mx = Array(state.Mx); My = Array(state.My); E = Array(state.E)
    V = Array(mesh.volume)
    return (; mass = sum(D .* V),
            mx = sum(Mx .* V),
            my = sum(My .* V),
            energy = sum(E .* V))
end

function max_signal_speed_2d(state::EulerState2D; gamma::Real)
    prim = conserved_to_primitive_2d(state; gamma)
    c = sqrt.(gamma .* max.(prim.pressure, 0) ./ prim.rho)
    return maximum(sqrt.(prim.vx .* prim.vx .+ prim.vy .* prim.vy) .+ c)
end
