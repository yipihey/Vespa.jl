#!/usr/bin/env python3

import argparse
import os
import shutil
import subprocess
from pathlib import Path

GAMMA = 5.0 / 3.0
BOX = 1.0
TIME_MAX = 0.045


def read_metadata(path):
    meta = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            key, value = line.strip().split(",", 1)
            meta[key] = value
    return meta


def read_table(path):
    return {"source": Path(path)}


def compile_c_writer(dest):
    src = Path(__file__).with_name("csv_to_arepo_ic.c")
    exe = dest / "csv_to_arepo_ic"
    if exe.exists() and exe.stat().st_mtime >= src.stat().st_mtime:
        return exe
    subprocess.run(["h5cc", str(src), "-O2", "-o", str(exe)], check=True)
    return exe


def write_ic(path, table, c_writer=None):
    if c_writer is None:
        raise RuntimeError("no compiled C HDF5 writer was provided")
    subprocess.run([str(c_writer), str(table["source"]), str(path)], check=True)


def config_text(local_ppm, shock_following=False, shock_gain=None, shock_width=None, shock_rmin=None):
    lines = [
        "#!/bin/bash",
        "TWODIMS",
        f"GAMMA={GAMMA:.16g}",
        "REGULARIZE_MESH_CM_DRIFT",
        "REGULARIZE_MESH_CM_DRIFT_USE_SOUNDSPEED",
        "REGULARIZE_MESH_FACE_ANGLE",
        "FORCE_EQUAL_TIMESTEPS",
        "DOUBLEPRECISION=1",
        "INPUT_IN_DOUBLEPRECISION",
        "OUTPUT_IN_DOUBLEPRECISION",
        "OUTPUT_CENTER_OF_MASS",
        "OUTPUT_VOLUME",
        "OUTPUT_PRESSURE",
        "OUTPUT_VERTEX_VELOCITY",
        "HAVE_HDF5",
    ]
    if local_ppm:
        lines.insert(3, "LOCAL_PPM")
    if shock_following:
        lines.append("SHOCK_FOLLOWING_MESH")
        if shock_gain is not None:
            lines.append(f"SHOCK_FOLLOWING_MESH_GAIN={shock_gain:.16g}")
        if shock_width is not None:
            lines.append(f"SHOCK_FOLLOWING_MESH_WIDTH={shock_width:.16g}")
        if shock_rmin is not None:
            lines.append(f"SHOCK_FOLLOWING_MESH_RMIN={shock_rmin:.16g}")
    return "\n".join(lines) + "\n"


def param_text(time_max=TIME_MAX, output_list_on=False):
    output_list = 1 if output_list_on else 0
    time_bet_snapshot = 0.0 if output_list_on else time_max
    return f"""InitCondFile                          ./IC
ICFormat                              3

OutputDir                             ./output/
SnapshotFileBase                      snap
SnapFormat                            3
NumFilesPerSnapshot                   1
NumFilesWrittenInParallel             1

OutputListOn                          {output_list}
OutputListFilename                    output_list.txt
ResubmitOn                            0
ResubmitCommand                       none
CoolingOn                             0
StarformationOn                       0

Omega0                                0.0
OmegaBaryon                           0.0
OmegaLambda                           0.0
HubbleParam                           1.0

BoxSize                               {BOX}
PeriodicBoundariesOn                  1
ComovingIntegrationOn                 0

MaxMemSize                            2500

TimeOfFirstSnapshot                   0.0
CpuTimeBetRestartFile                 9000
TimeLimitCPU                          90000

TimeBetStatistics                     1.0
TimeBegin                             0.0
TimeMax                               {time_max}
TimeBetSnapshot                       {time_bet_snapshot}

UnitVelocity_in_cm_per_s              1.0
UnitLength_in_cm                      1.0
UnitMass_in_g                         1.0
GravityConstantInternal               0.0

ErrTolIntAccuracy                     0.1
ErrTolTheta                           0.1
ErrTolForceAcc                        0.1

MaxSizeTimestep                       0.002
MinSizeTimestep                       1e-8
CourantFac                            0.3

CellShapingSpeed                      0.5
CellMaxAngleFactor                    2.25

LimitUBelowThisDensity                0.0
LimitUBelowCertainDensityToThisValue  0.0

DesNumNgb                             64
MultipleDomains                       2
TopNodeFactor                         4
ActivePartFracForNewDomainDecomp      0.01
MaxNumNgbDeviation                    2

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
"""


def write_case(
    dest,
    label,
    table,
    local_ppm,
    c_writer=None,
    time_max=TIME_MAX,
    output_times=None,
    shock_following=False,
    shock_gain=None,
    shock_width=None,
    shock_rmin=None,
):
    method = "localppm" if local_ppm else "muscl"
    suffix = "-shockfollow" if shock_following else ""
    case = dest / f"{label}{suffix}-{method}"
    case.mkdir(parents=True, exist_ok=True)
    output = case / "output"
    if output.exists():
        shutil.rmtree(output)
    output.mkdir()
    write_ic(case / "IC.hdf5", table, c_writer=c_writer)
    times = output_times or []
    (case / "param.txt").write_text(param_text(time_max=time_max, output_list_on=bool(times)), encoding="utf-8")
    (case / "output_list.txt").write_text(
        "".join(f"{t:.17g} 1\n" for t in times),
        encoding="utf-8",
    )
    (case / "Config.sh").write_text(
        config_text(
            local_ppm,
            shock_following=shock_following,
            shock_gain=shock_gain,
            shock_width=shock_width,
            shock_rmin=shock_rmin,
        ),
        encoding="utf-8",
    )
    return case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", required=True, type=Path)
    parser.add_argument("--dest", required=True, type=Path)
    parser.add_argument("--arepo", default="/Users/tabel/Projects/arepo", type=Path)
    parser.add_argument("--time-max", default=TIME_MAX, type=float)
    parser.add_argument(
        "--output-times",
        default="",
        help="Comma-separated snapshot times. Empty keeps the single TimeBetSnapshot output.",
    )
    parser.add_argument(
        "--shock-following-mesh",
        action="store_true",
        help="Also write cases with the experimental solution-following mesh velocity hook enabled.",
    )
    parser.add_argument("--shock-gain", default=None, type=float)
    parser.add_argument("--shock-width", default=None, type=float)
    parser.add_argument("--shock-rmin", default=None, type=float)
    args = parser.parse_args()
    output_times = [float(x) for x in args.output_times.split(",") if x.strip()]

    meta = read_metadata(args.tables / "metadata.txt")
    args.dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.tables / "metadata.txt", args.dest / "metadata.txt")
    c_writer = compile_c_writer(args.dest)

    cases = []
    for label in ("standard", "semi"):
        table = read_table(args.tables / meta[f"{label}_table"])
        for local_ppm in (False, True):
            cases.append(
                write_case(
                    args.dest,
                    label,
                    table,
                    local_ppm,
                    c_writer=c_writer,
                    time_max=args.time_max,
                    output_times=output_times,
                )
            )
            if args.shock_following_mesh:
                cases.append(
                    write_case(
                        args.dest,
                        label,
                        table,
                        local_ppm,
                        c_writer=c_writer,
                        time_max=args.time_max,
                        output_times=output_times,
                        shock_following=True,
                        shock_gain=args.shock_gain,
                        shock_width=args.shock_width,
                        shock_rmin=args.shock_rmin,
                    )
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

    print(f"wrote {len(cases)} AREPO cases under {args.dest}")
    print(f"build/run helper: {build}")


if __name__ == "__main__":
    main()
