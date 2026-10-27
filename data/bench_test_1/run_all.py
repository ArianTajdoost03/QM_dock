import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def call(script, *items):
    command = [sys.executable, os.path.join(HERE, script), *items]
    print("\n>>> " + " ".join(command), flush=True)
    code = subprocess.run(command).returncode
    if code:
        sys.exit(code)


def main():
    parser = argparse.ArgumentParser(description="Dock all complexes, compute RMSDs and run the analysis in one go")
    parser.add_argument("--data", required=True)
    parser.add_argument("--qmdock")
    parser.add_argument("--out", default="runs/base")
    parser.add_argument("--options", default=os.path.join(HERE, "dock_options.txt"))
    parser.add_argument("--opts")
    parser.add_argument("--ids", default=os.path.join(HERE, "ids_213.txt"))
    parser.add_argument("--state-select", choices=("corrected", "pooled-corrected", "energy", "pooled", "first"), default="pooled-corrected")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--baselines")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-docking", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args()

    out = os.path.abspath(args.out)
    if not args.skip_docking and not args.skip_preflight:
        call("check_inputs.py", "--data", args.data, "--ids", args.ids, "--out", out)
    if not args.skip_docking:
        dock = ["--data", args.data, "--out", out, "--ids", args.ids,
                "--jobs", str(args.jobs), "--timeout", str(args.timeout), "--limit", str(args.limit)]
        if args.qmdock:
            dock += ["--qmdock", args.qmdock]
        dock += ["--opts=" + args.opts] if args.opts else ["--options", args.options]
        if args.force:
            dock.append("--force")
        call("run_benchmark.py", *dock)
    call("collect_rmsd.py", "--runs", out, "--data", args.data, "--ids", args.ids, "--state-select", args.state_select)
    analyze = [os.path.join(out, "rmsd_results.csv"), "--out", os.path.join(out, "analysis")]
    if args.baselines:
        analyze += ["--baselines", args.baselines]
    call("analyze_benchmark.py", *analyze)
    call("attribute_failures.py", "--runs", out, "--data", args.data, "--ids", args.ids)


if __name__ == "__main__":
    main()
