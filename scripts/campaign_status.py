#!/usr/bin/env python
"""État d'avancement d'une campagne d'entraînement en cours.

Usage:
    uv run python scripts/campaign_status.py [--runs-root runs] [--expected 21]

Une campagne dure plusieurs jours et n'écrit son CSV agrégé qu'à la toute fin :
ce tableau de bord lit donc directement les répertoires d'exécution, pour
répondre aux trois questions qu'on se pose pendant l'attente — combien de runs
sont faites, où en est celle qui tourne, et quand la campagne finira.

L'estimation de fin repose sur la durée des runs déjà terminées quand il y en a,
et sur le débit observé de la run courante sinon.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def _fmt_duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    if days:
        return f"{days} j {hours:02d} h {minutes:02d} min"
    return f"{hours:02d} h {minutes:02d} min"


def _read_jsonl_tail(path: Path, kind: str) -> dict | None:
    """Dernière entrée de type ``kind`` d'un history.jsonl."""
    last = None
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if f'"{kind}"' not in line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:  # ligne en cours d'écriture
                    continue
                if entry.get("type") == kind:
                    last = entry
    except OSError:
        return None
    return last


def _gpu_state() -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    util, temp, used, total = (v.strip() for v in out.stdout.strip().split(","))
    return f"{util} % · {temp} °C · {used} / {total} MiB"


def _campaign_running() -> bool:
    try:
        out = subprocess.run(
            ["pgrep", "-f", "run_volume_curve"], capture_output=True, timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return out.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument(
        "--expected", type=int, default=None,
        help="nombre de runs attendues (pour l'estimation de fin)",
    )
    args = parser.parse_args()

    run_dirs = sorted(d for d in args.runs_root.glob("*/*") if d.is_dir())
    if not run_dirs:
        raise SystemExit(f"aucun répertoire de run sous {args.runs_root}")

    done, running = [], []
    for run_dir in run_dirs:
        summary_path = run_dir / "summary.json"
        history_path = run_dir / "history.jsonl"
        if summary_path.exists():
            done.append((run_dir, json.loads(summary_path.read_text(encoding="utf-8"))))
        elif history_path.exists():
            running.append((run_dir, history_path))

    print(f"campagne {'EN COURS' if _campaign_running() else 'ARRÊTÉE'}"
          f"{'' if _campaign_running() else ' — aucun processus run_volume_curve'}")
    gpu = _gpu_state()
    if gpu:
        print(f"GPU      {gpu}")
    print()

    total = args.expected or (len(done) + len(running))
    print(f"terminées : {len(done)} / {total}")
    mean_run = None
    if done:
        durations = [s["elapsed_seconds"] for _, s in done]
        mean_run = sum(durations) / len(durations)
        print(f"durée moyenne par run : {_fmt_duration(mean_run)}")
        for run_dir, summary in done[-5:]:
            print(f"  ✓ {run_dir.parent.name}/{run_dir.name:<16} "
                  f"IoU {summary['best_val_iou']:.4f} "
                  f"(iter {summary['best_iteration']}) · {_fmt_duration(summary['elapsed_seconds'])}")
    print()

    remaining_seconds = 0.0
    for run_dir, history_path in running:
        train = _read_jsonl_tail(history_path, "train")
        val = _read_jsonl_tail(history_path, "val")
        if not train:
            continue
        config_path = run_dir / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        target = config["iterations"]
        current = train["iteration"]
        # config.json est écrit une seule fois, juste avant l'entraînement : son
        # mtime est le début de la run. Celui de history.jsonl ne conviendrait
        # pas, le fichier étant réécrit à chaque itération.
        started = config_path.stat().st_mtime
        elapsed = time.time() - started
        rate = elapsed / current if current else 0.0

        print(f"en cours : {run_dir.parent.name}/{run_dir.name}")
        pct = 100.0 * current / target
        bar = "█" * int(pct // 4) + "·" * (25 - int(pct // 4))
        print(f"  {bar} {current} / {target} ({pct:.1f} %)")
        print(f"  perte   {train['loss']:.4f} · {rate:.2f} s/itération")
        if val:
            print(f"  dernière validation (iter {val['iteration']}) : "
                  f"IoU dommage {val['mean_iou_damage']:.4f} · ECE {val['ece']:.4f}")
        left = (target - current) * rate
        remaining_seconds += left
        print(f"  fin de cette run dans ~{_fmt_duration(left)}")
        print()

    queued = max(0, total - len(done) - len(running))
    if queued and (mean_run or running):
        per_run = mean_run or ((target * rate) if running else 0)
        remaining_seconds += queued * per_run
        print(f"en attente : {queued} run(s)")
    if remaining_seconds:
        end = time.localtime(time.time() + remaining_seconds)
        print(f"fin estimée de la campagne : dans ~{_fmt_duration(remaining_seconds)} "
              f"({time.strftime('%a %d/%m à %H:%M', end)})")


if __name__ == "__main__":
    main()
