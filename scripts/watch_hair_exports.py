#!/usr/bin/env python3
"""Export completed cases from a running Cycloud SMHC batch, using only CPU.

The watcher runs independently of the GPU worker so it also handles its last case
after the GPU job has exited. Existing local results excluded from the GPU batch
are included when their complete final arrays are present.
"""

import argparse
import fcntl
import json
import os
import signal
import time
from pathlib import Path

from export_hair import INPUT_FILES, export, timestamp

STOP = False


def atomic_json(path, data):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--hairutil", default="hairutil")
    parser.add_argument("--poll-seconds", type=float, default=10)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    batch = args.batch_dir.resolve()
    manifest = json.loads((batch / "manifest.json").read_text())
    # Keep exports next to a batch moved together with its output tree.
    output = (
        batch.parent.parent
        if batch.parent.name == "_cycloud"
        else Path(manifest["output_root"])
    )
    cloud_cases = set(manifest["cases"])
    local_cases = [
        case
        for case in manifest.get("excluded_cases", [])
        if all(
            (output / case / "multiview_hairgrowing" / name).is_file()
            for name in INPUT_FILES
        )
    ]
    cases = sorted(cloud_cases.union(local_cases))
    state = {
        "status": "RUNNING",
        "started_at": timestamp(),
        "host": os.uname().nodename,
        "pid": os.getpid(),
        "total": len(cases),
        "poll_seconds": args.poll_seconds,
        "cases": [
            {"case": case, "status": "WAITING_FOR_RECONSTRUCTION"} for case in cases
        ],
    }

    def save():
        state["updated_at"] = timestamp()
        state["converted"] = sum(c["status"] == "SUCCEEDED" for c in state["cases"])
        state["failed"] = sum(
            c["status"] in ("FAILED", "RECONSTRUCTION_FAILED") for c in state["cases"]
        )
        atomic_json(batch / "export_status.json", state)

    def stop(signum, frame):
        global STOP
        STOP = True

    with (batch / "export_watcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        (batch / "export_watcher.pid").write_text(str(os.getpid()) + "\n")
        save()
        print(
            f"[{timestamp()}] Watching {len(cases)} cases; no GPU required", flush=True
        )
        terminal_since = None
        try:
            while not STOP:
                reconstruction = json.loads((batch / "status.json").read_text())
                states = {c["case"]: c["status"] for c in reconstruction["cases"]}
                terminal = reconstruction["status"] in (
                    "SUCCEEDED",
                    "FAILED",
                    "INTERRUPTED",
                )
                if terminal and terminal_since is None:
                    terminal_since = time.monotonic()
                elif not terminal:
                    terminal_since = None
                # NFS may expose status.json before a newly created completion
                # marker because negative directory lookups remain cached.
                terminal_settled = terminal and time.monotonic() - terminal_since >= 60
                state["reconstruction_status"] = reconstruction["status"]
                for entry in state["cases"]:
                    if STOP:
                        break
                    case = entry["case"]
                    if entry["status"] == "SUCCEEDED":
                        continue
                    directory = output / case / "multiview_hairgrowing"
                    marker_path = output / case / "cycloud_result.json"
                    ready = case in local_cases
                    if case in cloud_cases and marker_path.is_file():
                        marker = json.loads(marker_path.read_text())
                        ready = (
                            marker.get("status") == "SUCCEEDED"
                            and marker.get("job_name") == manifest["job_name"]
                        )
                    if not ready:
                        if terminal_settled or states.get(case) == "FAILED":
                            entry["status"] = "RECONSTRUCTION_FAILED"
                            entry["error"] = (
                                "Reconstruction has no validated completion marker"
                            )
                        continue
                    if entry.get("attempts", 0) >= 3:
                        continue
                    entry.update(
                        {
                            "status": "CONVERTING",
                            "attempts": entry.get("attempts", 0) + 1,
                        }
                    )
                    save()
                    try:
                        result = export(directory, args.hairutil)
                        entry.update(
                            {
                                "status": "SUCCEEDED",
                                "finished_at": result["finished_at"],
                                "points": result["points"],
                                "strands": result["strands"],
                                "data": str(directory / "final.data"),
                                "ply": str(directory / "final.ply"),
                            }
                        )
                        entry.pop("error", None)
                        print(
                            f"[{timestamp()}] EXPORTED {case}: {result['strands']} strands / {result['points']} points",
                            flush=True,
                        )
                    except Exception as error:
                        entry.update({"status": "FAILED", "error": str(error)})
                        print(
                            f"[{timestamp()}] Export failed for {case}: {error}",
                            flush=True,
                        )
                    save()
                save()
                if state["converted"] == len(cases):
                    state["status"] = "SUCCEEDED"
                    break
                retry_pending = any(
                    c["status"] == "FAILED" and c.get("attempts", 0) < 3
                    for c in state["cases"]
                )
                publication_pending = any(
                    c["status"] == "WAITING_FOR_RECONSTRUCTION" for c in state["cases"]
                )
                if terminal and not retry_pending and not publication_pending:
                    state["status"] = "FAILED"
                    break
                time.sleep(args.poll_seconds)
            if STOP:
                state["status"] = "INTERRUPTED"
        except Exception as error:
            state.update({"status": "FAILED", "error": str(error)})
            raise
        finally:
            state["finished_at"] = timestamp()
            save()
            print(
                f"[{timestamp()}] Watcher {state['status']}: {state['converted']}/{len(cases)} exported",
                flush=True,
            )
    return 0 if state["status"] == "SUCCEEDED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
