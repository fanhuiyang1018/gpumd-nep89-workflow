#!/usr/bin/env python3
"""Create temperature-specific GPUMD run folders from run.in and model.xyz."""
import argparse
import re
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create temperature folders and rewrite the npt_scr temperature fields in run.in."
    )
    parser.add_argument(
        "-T", "--temps", nargs="+", type=int,
        help="Target temperatures, for example: -T 400 500 600. Defaults to 300-1500 K with a 200 K step."
    )
    return parser.parse_args()


def get_temperatures(args: argparse.Namespace) -> list[int]:
    if args.temps:
        return sorted(args.temps)
    return list(range(300, 1501, 200))


def main() -> None:
    workdir = Path.cwd()
    src_run = workdir / "run.in"
    src_model = workdir / "model.xyz"

    for required_file in (src_run, src_model):
        if not required_file.is_file():
            raise SystemExit(f"Missing {required_file.name}; aborting.")

    for temperature in get_temperatures(parse_args()):
        folder = workdir / f"{temperature}K"
        folder.mkdir(exist_ok=True)

        text = src_run.read_text(encoding="utf-8")
        text = re.sub(
            r"^(\s*ensemble\s+npt_scr\s+)\d+\s+\d+",
            rf"\g<1>{temperature} {temperature}",
            text,
            flags=re.M,
        )

        (folder / "run.in").write_text(text, encoding="utf-8")
        shutil.copy2(src_model, folder / "model.xyz")
        print(f"Prepared {folder.name}")

    print("All temperature folders have been prepared.")


if __name__ == "__main__":
    main()
