#!/usr/bin/env python3

import argparse
import os
import shutil
import subprocess
from pathlib import Path

GAMMA = 5.0 / 3.0
TIME_MAX = 2.0


def read_metadata(path):
    meta = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            key, value = line.strip().split(",", 1)
            meta[key] = value
    return meta


def compile_c_writer(dest):
    src = Path(__file__).parents[1] / "arepo_sedov_proxy" / "csv_to_arepo_ic.c"
    exe = dest / "csv_to_arepo_ic"
    if exe.exists() and exe.stat().st_mtime >= src.stat().st_mtime:
        return exe
    subprocess.run(["h5cc", str(src), "-O2", "-o", str(exe)], check=True)
    return exe


def write_ic(path, table, box_size, c_writer):
    subprocess.run([str(c_writer), str(table), str(path), f"{box_size:.17g}"], check=True)


def config_text(local_ppm, hll=False, bulk_viscosity=False,
                bulk_viscosity_quad=1.0, bulk_viscosity_linear=0.0,
                bulk_viscosity_pressure_jump=0.05,
                bulk_viscosity_pressure_cap=10.0):
    lines = [
        "#!/bin/bash",
        "TWODIMS",
        "REFLECTIVE_X=2",
        "REFLECTIVE_Y=2",
        f"GAMMA={GAMMA:.16g}",
        "REGULARIZE_MESH_CM_DRIFT",
        "REGULARIZE_MESH_CM_DRIFT_USE_SOUNDSPEED",
        "REGULARIZE_MESH_FACE_ANGLE",
        "TREE_BASED_TIMESTEPS",
        "DOUBLEPRECISION=1",
        "INPUT_IN_DOUBLEPRECISION",
        "OUTPUT_IN_DOUBLEPRECISION",
        "OUTPUT_CENTER_OF_MASS",
        "OUTPUT_VOLUME",
        "OUTPUT_PRESSURE",
        "HAVE_HDF5",
    ]
    if local_ppm:
        lines.insert(5, "LOCAL_PPM")
    if hll:
        insert_at = 6 if local_ppm else 5
        lines.insert(insert_at, "RIEMANN_HLL")
    if bulk_viscosity:
        insert_at = 7 if local_ppm and hll else 6 if (local_ppm or hll) else 5
        lines.insert(insert_at, "ARTIFICIAL_BULK_VISCOSITY")
        lines.insert(insert_at + 1, f"ARTIFICIAL_BULK_VISCOSITY_QUAD={bulk_viscosity_quad:.16g}")
        lines.insert(insert_at + 2, f"ARTIFICIAL_BULK_VISCOSITY_LINEAR={bulk_viscosity_linear:.16g}")
        lines.insert(insert_at + 3, f"ARTIFICIAL_BULK_VISCOSITY_PRESSURE_JUMP={bulk_viscosity_pressure_jump:.16g}")
        lines.insert(insert_at + 4, f"ARTIFICIAL_BULK_VISCOSITY_PRESSURE_CAP={bulk_viscosity_pressure_cap:.16g}")
    return "\n".join(lines) + "\n"


def param_text(box_size, time_max=TIME_MAX, output_times=None):
    output_times = output_times or []
    output_list = 1 if output_times else 0
    time_bet_snapshot = 0.5 if not output_times else 0.0
    return f"""InitCondFile                          ./IC
ICFormat                              3

OutputDir                             ./output/
SnapshotFileBase                      snap
SnapFormat                            3
NumFilesPerSnapshot                   1
NumFilesWrittenInParallel             1

ResubmitOn                            0
ResubmitCommand                       none
OutputListFilename                    output_list.txt
OutputListOn                          {output_list}

CoolingOn                             0
StarformationOn                       0

Omega0                                0.0
OmegaBaryon                           0.0
OmegaLambda                           0.0
HubbleParam                           1.0

BoxSize                               {box_size:.17g}
PeriodicBoundariesOn                  1
ComovingIntegrationOn                 0

MaxMemSize                            2500

TimeOfFirstSnapshot                   0.0
CpuTimeBetRestartFile                 9000
TimeLimitCPU                          90000

TimeBetStatistics                     0.005
TimeBegin                             0.0
TimeMax                               {time_max:.17g}
TimeBetSnapshot                       {time_bet_snapshot:.17g}

UnitVelocity_in_cm_per_s              1.0
UnitLength_in_cm                      1.0
UnitMass_in_g                         1.0
GravityConstantInternal               0.0

ErrTolIntAccuracy                     0.1
ErrTolTheta                           0.1
ErrTolForceAcc                        0.1

MaxSizeTimestep                       0.2
MinSizeTimestep                       1e-5
CourantFac                            0.3

LimitUBelowThisDensity                0.0
LimitUBelowCertainDensityToThisValue  0.0
DesNumNgb                             64
MaxNumNgbDeviation                    2

MultipleDomains                       2
TopNodeFactor                         4
ActivePartFracForNewDomainDecomp      0.5

TypeOfTimestepCriterion               0
TypeOfOpeningCriterion                1
GasSoftFactor                         0.01

SofteningComovingType0                0.1
SofteningComovingType1                0.1
SofteningComovingType2                0.1
SofteningComovingType3                0.1
SofteningComovingType4                0.1
SofteningComovingType5                0.1

SofteningMaxPhysType0                 0.1
SofteningMaxPhysType1                 0.1
SofteningMaxPhysType2                 0.1
SofteningMaxPhysType3                 0.1
SofteningMaxPhysType4                 0.1
SofteningMaxPhysType5                 0.1

SofteningTypeOfPartType0              0
SofteningTypeOfPartType1              0
SofteningTypeOfPartType2              0
SofteningTypeOfPartType3              0
SofteningTypeOfPartType4              0
SofteningTypeOfPartType5              0

InitGasTemp                           0.0
MinGasTemp                            0.0
MinEgySpec                            0.0
MinimumDensityOnStartUp               0.0

CellShapingSpeed                      0.5
CellMaxAngleFactor                    2.25
"""


def write_case(dest, label, table, local_ppm, hll, box_size, c_writer, time_max, output_times,
               bulk_viscosity, bulk_viscosity_quad, bulk_viscosity_linear,
               bulk_viscosity_pressure_jump, bulk_viscosity_pressure_cap):
    method = "localppm" if local_ppm else "muscl"
    suffix = "-hll" if hll else ""
    case = dest / f"{label}-{method}{suffix}"
    case.mkdir(parents=True, exist_ok=True)
    output = case / "output"
    if output.exists():
        shutil.rmtree(output)
    output.mkdir()
    write_ic(case / "IC.hdf5", table, box_size, c_writer)
    (case / "param.txt").write_text(
        param_text(box_size, time_max=time_max, output_times=output_times),
        encoding="utf-8",
    )
    (case / "output_list.txt").write_text(
        "".join(f"{t:.17g} 1\n" for t in output_times),
        encoding="utf-8",
    )
    (case / "Config.sh").write_text(
        config_text(local_ppm, hll=hll, bulk_viscosity=bulk_viscosity,
                    bulk_viscosity_quad=bulk_viscosity_quad,
                    bulk_viscosity_linear=bulk_viscosity_linear,
                    bulk_viscosity_pressure_jump=bulk_viscosity_pressure_jump,
                    bulk_viscosity_pressure_cap=bulk_viscosity_pressure_cap),
        encoding="utf-8",
    )
    return case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", required=True, type=Path)
    parser.add_argument("--dest", required=True, type=Path)
    parser.add_argument("--arepo", default="/Users/tabel/Projects/arepo", type=Path)
    parser.add_argument("--time-max", default=TIME_MAX, type=float)
    parser.add_argument("--output-times", default="0.5,1.0,1.5,2.0")
    parser.add_argument("--include-hll", action="store_true")
    parser.add_argument("--bulk-viscosity", action="store_true")
    parser.add_argument("--bulk-viscosity-quad", default=1.0, type=float)
    parser.add_argument("--bulk-viscosity-linear", default=0.0, type=float)
    parser.add_argument("--bulk-viscosity-pressure-jump", default=0.05, type=float)
    parser.add_argument("--bulk-viscosity-pressure-cap", default=10.0, type=float)
    args = parser.parse_args()
    output_times = [float(x) for x in args.output_times.split(",") if x.strip()]

    meta = read_metadata(args.tables / "metadata.txt")
    box_size = float(meta["box"])
    args.dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.tables / "metadata.txt", args.dest / "metadata.txt")
    c_writer = compile_c_writer(args.dest)

    cases = []
    for label in ("standard", "powerfoam"):
        table = args.tables / meta[f"{label}_table"]
        for local_ppm in (False, True):
            for hll in ((False, True) if args.include_hll else (False,)):
                cases.append(
                    write_case(args.dest, label, table, local_ppm, hll, box_size,
                               c_writer, args.time_max, output_times,
                               args.bulk_viscosity, args.bulk_viscosity_quad,
                               args.bulk_viscosity_linear,
                               args.bulk_viscosity_pressure_jump,
                               args.bulk_viscosity_pressure_cap)
                )

    build = args.dest / "build_and_run.sh"
    with build.open("w", encoding="utf-8") as handle:
        handle.write("#!/bin/bash\nset -euo pipefail\n\n")
        for case in cases:
            name = case.name
            exe = case / "Arepo"
            bdir = args.dest / f"build-{name}"
            handle.write(
                f"make -C {args.arepo} CONFIG={case / 'Config.sh'} "
                f"BUILD_DIR={bdir} EXEC={exe}\n"
            )
            handle.write(f"(cd {case} && mpiexec -np 1 ./Arepo param.txt)\n\n")
    os.chmod(build, 0o755)

    print(f"wrote {len(cases)} AREPO Noh cases under {args.dest}")
    print(f"build/run helper: {build}")


if __name__ == "__main__":
    main()
