"""Progress tracking for long runs.

output/progress.log  one line per finished unit of work (append)
output/progress.txt  the current status only (overwritten)

A unit is one LightGBM fit, or one batch of SARIMA origins. The ETA assumes
the remaining units take as long as the finished ones on average.
"""
import time
from pathlib import Path

_state = {"total": 0, "done": 0, "t0": None, "stage": "", "dir": None}


def start(total: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    _state.update(total=total, done=0, t0=time.time(), stage="", dir=out_dir)
    (out_dir / "progress.log").write_text(
        f"{time.strftime('%Y-%m-%d %H:%M:%S')}  run started · {total} units planned\n", encoding="utf-8")
    _write_status("starting")


def stage(name: str):
    _state["stage"] = name
    if _state["dir"]:
        _append(f"--- {name}")


def tick(label: str = "", units: int = 1):
    if _state["t0"] is None:
        return
    _state["done"] += units
    line = f"{_summary()} · {_state['stage']} {label}".rstrip()
    _append(line)
    _write_status(line)


def finish():
    if _state["t0"] is None:
        return
    line = f"run finished · {_fmt(time.time() - _state['t0'])} total"
    _append(line)
    _write_status(line)


def _summary():
    done, total = _state["done"], max(_state["total"], 1)
    el = time.time() - _state["t0"]
    eta = el / done * max(total - done, 0) if done else float("nan")
    return f"{done}/{total} ({100 * done / total:.0f}%) · elapsed {_fmt(el)} · ETA ~{_fmt(eta)}"


def _fmt(s):
    if s != s:
        return "?"
    s = int(s)
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def _append(line):
    with open(_state["dir"] / "progress.log", "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%H:%M:%S')}  {line}\n")


def _write_status(line):
    (_state["dir"] / "progress.txt").write_text(
        f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n{line}\n", encoding="utf-8")
