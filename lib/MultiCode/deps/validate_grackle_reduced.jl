# Standalone validation of the reduced-chemistry service: one-zone primordial
# evolution z=1000 -> 10 (the same setup as the Grackle fork's cosmo_evolve.c
# HE=1 run), confirming the code-neutral service reproduces x_HII and x_H2 with
# only HII and H2I advected.  This is the shared chemistry every MultiCode host
# (RAMSES, Arepo) will call.
include(joinpath(@__DIR__, "..", "src", "grackle_service.jl"))
using .GrackleChem
using Printf

const mh = 1.67262171e-24
const XH = 0.76
cosmic_time(z, H0_s, Om) = (2/3)/H0_s/sqrt(Om) * (1+z)^(-1.5)

h=0.71; Om=0.27; OL=0.73; Ob=0.046
H0_s = h*100*1e5/3.0857e24
rho_crit0 = 1.8788e-29*h*h
nH0 = Ob*rho_crit0*XH/mh                       # mean n_H today [cm^-3]

# code units: density in m_H, length 1 kpc, time 1 Myr (as cosmo_evolve.c)
dens_u = mh; len_u = 3.0857e21; time_u = 3.1557e13
GD = get(ENV,"GRACKLE_DATA_FILE",
         joinpath(homedir(),"Research","codes","grackle","input","CloudyData_noUVB.h5"))

z = 1000.0
GrackleChem.grackle_reduced_init!(; hubble=h*100, Om=Om, OL=OL, a_value=1/(1+z),
    fh=XH, density_units=dens_u, length_units=len_u, time_units=time_u, data_file=GD)

# RECFAST ICs at z=1000
T = 2728.0; xe = 0.047
nH = nH0*(1+z)^3; rho_tot = nH*mh/XH
rho  = [rho_tot/dens_u]                          # code density
HII  = [xe*XH*rho[1]]                             # rho*x_HII
H2I  = [2e-15*XH*rho[1]]                          # rho*x_H2 (mass), seed 1e-15
# specific internal energy in code units: e = T/(Tunits)/(gamma-1)/mu
vel_u = len_u/time_u; Tunits = mh*vel_u^2/1.380649e-16  # m_H v^2 / k_B
e_int = [T/Tunits/(5/3-1)/1.22]

println("# z      x_HII       x_H2        T_gas")
dlna = 0.01
while z > 10.0
    znew = (1+z)*exp(-dlna) - 1; znew < 10 && (znew = 10.0)
    fac = (1+znew)/(1+z); f3 = fac^3
    rho[1]*=f3; HII[1]*=f3; H2I[1]*=f3; e_int[1]*=fac^2     # dilute + adiabatic
    dt = (cosmic_time(znew,H0_s,Om) - cosmic_time(z,H0_s,Om))/time_u
    GrackleChem.grackle_reduced_step!(rho, e_int, HII, H2I; a_value=1/(1+znew), dt=dt)
    global z = znew
end
Tg = GrackleChem.grackle_reduced_temperature(rho, e_int, HII, H2I; a_value=1/(1+z))
nHmass = XH*rho[1]
@printf("z=%6.2f  x_HII=%.4e  x_H2=%.4e  T_gas=%.3f K\n",
        z, HII[1]/nHmass, (H2I[1]/2)/nHmass, Tg[1])
println("(Enzo cosmo_evolve HE=1 reference at z=10: x_HII~2.0e-4, x_H2~3.1e-6)")
