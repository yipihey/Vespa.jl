# Grackle reduced-chemistry WORKER — runs in its own process so the live Grackle
# never shares an address space with the host code (RAMSES).  In-process, RAMSES
# + Grackle co-residency segfaults inside solve_rate_cool_g (a Fortran-runtime /
# allocator interaction); the exact same arrays/units run fine here in isolation.
#
# Binary protocol over stdin→stdout (host is little-endian; worker matches):
#   init  : hubble,Om,OL,a0,fh,density_units,length_units,time_units (8×Float64),
#           deuterium,namelen (2×Int64), data_file bytes  →  Int64(1) ack
#   step  : Int64(1), n(Int64), a_value,dt (2×Float64),
#           rho[n],eint[n],HII[n],H2I[n](,HDI[n]) (Float64)
#           →  eint[n],HII[n],H2I[n](,HDI[n]) back
#   quit  : Int64(0)
# All chatter (precompile, @info) goes to stderr; stdout carries ONLY the protocol.

using MultiCode
const IN = stdin; const OUT = stdout

# Process n cells in ≤BATCH-cell Grackle calls (a single ~2M-cell call accumulates
# state in one tight chunk loop and segfaults; separate calls don't).  GC between.
function process_step!(rho, eint, HII, H2I, HDI, a, dt; BATCH=131072)
    n = length(rho); off = 0
    while off < n
        m = min(BATCH, n-off); rng = (off+1):(off+m)
        br = rho[rng]; be = eint[rng]; bh = HII[rng]; b2 = H2I[rng]
        bd = HDI === nothing ? nothing : HDI[rng]
        MultiCode.GrackleChem.grackle_reduced_step!(br, be, bh, b2, bd; a_value=a, dt=dt)
        eint[rng] = be; HII[rng] = bh; H2I[rng] = b2
        HDI === nothing || (HDI[rng] = bd)
        off += m; GC.gc(false)
    end
end

hubble=read(IN,Float64); Om=read(IN,Float64); OL=read(IN,Float64)
a0=read(IN,Float64); fh=read(IN,Float64)
du=read(IN,Float64); lu=read(IN,Float64); tu=read(IN,Float64)
deut=read(IN,Int64); flen=read(IN,Int64); data_file=String(read(IN,flen))

MultiCode.chem_init!(; hubble=hubble, Om=Om, OL=OL, a_value=a0, fh=fh,
    density_units=du, length_units=lu, time_units=tu, data_file=data_file,
    deuterium=(deut==1))
write(OUT, Int64(1)); flush(OUT)                       # ready

while !eof(IN)
    cmd = read(IN, Int64)
    cmd == 0 && break
    n = read(IN, Int64); a = read(IN, Float64); dt = read(IN, Float64)
    rho  = Vector{Float64}(undef,n); read!(IN, rho)
    eint = Vector{Float64}(undef,n); read!(IN, eint)
    HII  = Vector{Float64}(undef,n); read!(IN, HII)
    H2I  = Vector{Float64}(undef,n); read!(IN, H2I)
    HDI  = deut==1 ? (v=Vector{Float64}(undef,n); read!(IN,v); v) : nothing
    if get(ENV,"CHEM_DEBUG","0")=="1"
        println(stderr, "worker step: n=$n a=$a dt=$dt eint=$(extrema(eint)) ",
                "HII=$(extrema(HII)) H2I=$(extrema(H2I)) HDI=$(HDI===nothing ? "-" : extrema(HDI))"); flush(stderr)
    end
    process_step!(rho, eint, HII, H2I, HDI, a, dt)
    write(OUT, eint); write(OUT, HII); write(OUT, H2I)
    deut==1 && write(OUT, HDI)
    flush(OUT)
end
