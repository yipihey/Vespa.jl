# Low-memory Metal particle locality reorder.

using Random
using PoissonKernels

const _PARTICLE_SORT_IMPORTED_METAL = try
    @eval using Metal
    true
catch
    false
end

@inline function _sort_part1by2(n::UInt32)
    n &= 0x000003ff
    n = (n | (n << 16)) & 0xff0000ff
    n = (n | (n <<  8)) & 0x0300f00f
    n = (n | (n <<  4)) & 0x030c30c3
    n = (n | (n <<  2)) & 0x09249249
    return n
end
@inline _sort_morton3(x::UInt32, y::UInt32, z::UInt32) =
    _sort_part1by2(x) | (_sort_part1by2(y) << 1) | (_sort_part1by2(z) << 2)

function _sort_bucket_key(x, y, z, B::Integer)
    Bf = Float32(B); Bm1 = UInt32(B - 1)
    ix = min(unsafe_trunc(UInt32, floor(max(0f0, Float32(x) * Bf))), Bm1)
    iy = min(unsafe_trunc(UInt32, floor(max(0f0, Float32(y) * Bf))), Bm1)
    iz = min(unsafe_trunc(UInt32, floor(max(0f0, Float32(z) * Bf))), Bm1)
    return _sort_morton3(ix, iy, iz)
end

function _sort_metal_ready()
    try
        _PARTICLE_SORT_IMPORTED_METAL || return false
        Metal.functional() || return false
        for _ in 1:10
            PoissonKernels.has_backend(:metal) && return true
            sleep(0.05)
        end
        return false
    catch
        return false
    end
end

@testset "particle Morton reorder" begin
    rng = MersenneTwister(0x51caff)
    np = 4096
    N = 32
    B = 8
    px = rand(rng, Float32, np)
    py = rand(rng, Float32, np)
    pz = rand(rng, Float32, np)
    vx = Float16.(0.001f0 .* randn(rng, Float32, np))
    vy = Float16.(0.001f0 .* randn(rng, Float32, np))
    vz = Float16.(0.001f0 .* randn(rng, Float32, np))
    id = UInt32.(1:np)

    if _sort_metal_ready()
        be = PoissonKernels.backend(:metal)
        dparts = (
            px = PoissonKernels.to_device(be, px, Float32),
            py = PoissonKernels.to_device(be, py, Float32),
            pz = PoissonKernels.to_device(be, pz, Float32),
            vx = PoissonKernels.to_device(be, vx, Float16),
            vy = PoissonKernels.to_device(be, vy, Float16),
            vz = PoissonKernels.to_device(be, vz, Float16),
            mass = Float32(1),
            id = PoissonKernels.to_device(be, id, UInt32),
        )
        withenv("CIC_PSORT_MODE" => "bucket", "CIC_PSORT_BUCKET" => string(B)) do
            MultiCode.morton_sort_particles!(dparts; N)
        end
        hx = PoissonKernels.to_host(dparts.px)
        hy = PoissonKernels.to_host(dparts.py)
        hz = PoissonKernels.to_host(dparts.pz)
        hvx = PoissonKernels.to_host(dparts.vx)
        hvy = PoissonKernels.to_host(dparts.vy)
        hvz = PoissonKernels.to_host(dparts.vz)
        hid = PoissonKernels.to_host(dparts.id)
        hkey = [_sort_bucket_key(hx[i], hy[i], hz[i], B) for i in eachindex(hx)]
        @test issorted(hkey)
        @test sort(hid) == id
        @test all(hx[j] == px[Int(hid[j])] for j in eachindex(hid))
        @test all(hy[j] == py[Int(hid[j])] for j in eachindex(hid))
        @test all(hz[j] == pz[Int(hid[j])] for j in eachindex(hid))
        @test all(hvx[j] == vx[Int(hid[j])] for j in eachindex(hid))
        @test all(hvy[j] == vy[Int(hid[j])] for j in eachindex(hid))
        @test all(hvz[j] == vz[Int(hid[j])] for j in eachindex(hid))
    else
        @test_skip "Metal not available — bucket particle reorder skipped"
    end
end
