# Minimal double-precision Gadget format-1 IC writer (ICFormat=1, INPUT_IN_DOUBLEPRECISION).
# Blocks: header(256B) POS VEL ID MASS [U for gas], each wrapped in int32 Fortran record markers.
module GadgetIC
using Printf
function write_ic(path; pos, vel, ids, mass, u=Float64[], pass=zeros(0,0), ngas::Int=0,
                  boxsize, a, omega0, omegal, hubble, massarr=zeros(6))
    n = size(pos,1)
    @assert size(pos)==(n,3) && size(vel)==(n,3) && length(ids)==n && length(mass)==n
    npart = zeros(Int32,6); npart[1]=ngas; npart[2]=n-ngas      # type0=gas, type1=DM
    rec(io,nbytes) = write(io, Int32(nbytes))
    open(path,"w") do io
        # ── header (256 bytes) ──
        rec(io,256)
        for v in npart; write(io, Int32(v)); end
        for v in massarr; write(io, Float64(v)); end
        write(io, Float64(a)); write(io, Float64(1/a-1))         # time(a), redshift
        write(io, Int32(0), Int32(0))                            # flag_sfr, flag_feedback
        for v in npart; write(io, UInt32(v)); end                # npartTotal
        write(io, Int32(0), Int32(1))                            # flag_cooling, num_files
        write(io, Float64(boxsize), Float64(omega0), Float64(omegal), Float64(hubble))
        write(io, Int32(0), Int32(0))                            # flag_stellarage, flag_metals
        for _ in 1:6; write(io, UInt32(0)); end                  # npartTotalHighWord
        write(io, Int32(0))                                      # flag_entropy_instead_u
        write(io, Int32(1))                                      # flag_doubleprecision = 1
        write(io, Int32(0))                                      # flag_lpt_ics
        write(io, Float32(0))                                    # lpt_scalingfactor
        write(io, Int32(0))                                      # flag_tracer_field
        write(io, Int32(0))                                      # composition_vector_length
        for _ in 1:40; write(io, UInt8(0)); end                 # fill to 256
        rec(io,256)
        # ── POS (3n float64) ──
        rec(io, 24n); for p in 1:n, d in 1:3; write(io, Float64(pos[p,d])); end; rec(io,24n)
        # ── VEL ──
        rec(io, 24n); for p in 1:n, d in 1:3; write(io, Float64(vel[p,d])); end; rec(io,24n)
        # ── IDs (int32) ──
        rec(io, 4n); for p in 1:n; write(io, Int32(ids[p])); end; rec(io,4n)
        # ── MASS (only for types with massarr==0; here all) ──
        rec(io, 8n); for p in 1:n; write(io, Float64(mass[p])); end; rec(io,8n)
        # ── U (gas internal energy) ──
        if ngas>0
            rec(io, 8ngas); for p in 1:ngas; write(io, Float64(u[p])); end; rec(io,8ngas)
        end
        # ── PASS (gas passive scalars, ngas × nps float64) ──
        if ngas>0 && size(pass,1)==ngas && size(pass,2)>0
            nps = size(pass,2)
            rec(io, 8*nps*ngas)
            for p in 1:ngas, s in 1:nps; write(io, Float64(pass[p,s])); end
            rec(io, 8*nps*ngas)
        end
    end
    return path
end
end
