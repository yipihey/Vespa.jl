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

struct FaceFluxWork2D{A<:AbstractVector}
    FD::A
    FMx::A
    FMy::A
    FE::A
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

const _METAL_PKGID = Base.PkgId(Base.UUID("dde4c033-4e86-420c-a63e-0dd931031962"), "Metal")

function _backend_zeros(be, ::Type{T}, n::Integer) where {T}
    if occursin("MetalBackend", string(typeof(be))) && haskey(Base.loaded_modules, _METAL_PKGID)
        metal = Base.loaded_modules[_METAL_PKGID]
        storage_name = lowercase(get(ENV, "POWERFOAM_METAL_STORAGE", "shared"))
        storage = storage_name == "private" ?
                  getproperty(metal, :PrivateStorage) :
                  getproperty(metal, :SharedStorage)
        return Base.invokelatest(getproperty(metal, :zeros), T, n; storage)
    end
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

hydro_work_2d(state::EulerState2D, mesh::ArepoMeshArrays2D) =
    FaceFluxWork2D(similar(state.D, length(mesh.c1)),
                   similar(state.D, length(mesh.c1)),
                   similar(state.D, length(mesh.c1)),
                   similar(state.D, length(mesh.c1)))

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
