"""A43 / author: measure total throughput of the concurrent P1 location training jobs (their training, 128 px ResNet) at 4, 2
and 1 concurrent jobs, by pausing jobs with SIGSTOP (main process + DataLoader workers) and timing epochs from their
train.log ("epoch N Mean train loss" lines). Paused jobs resume immediately after; nothing else changes. Prints and writes
/workspace/logs/p1_loc_throughput.json: epochs per hour per running job and in total, per concurrency level.
    /workspace/venv-wtss/bin/python scripts/replication/p1_loc_measure.py [--epochs 2]
"""
import argparse, json, os, re, signal, subprocess, time
from pathlib import Path
D = os.environ.get("WTSS_DATA", "/workspace/data")


def jobs():
    out = []
    for l in subprocess.run(["ps", "-eo", "pid,ppid,args"], capture_output=True, text=True).stdout.splitlines()[1:]:
        pid, ppid, a = l.strip().split(None, 2)
        if "p1_location.py train" in a and "bash" not in a.split()[0]:
            out.append((int(pid), int(ppid), a))
    pids = {p for p, _, _ in out}
    mains = [(p, a) for p, pp, a in out if pp not in pids]
    res = []
    for p, a in mains:
        env = open(f"/proc/{p}/environ").read().split("\0"); s = [e for e in env if e.startswith("P1_SEED=")][0].split("=")[1]
        pos = re.search(r"--position (\w+)", a).group(1)
        log = Path(f"{D}/replication/viviano/location_s{s}/{pos}/train.log")
        kids = subprocess.run(["pgrep", "-P", str(p)], capture_output=True, text=True).stdout.split()
        res.append({"pid": p, "kids": [int(k) for k in kids], "seed": s, "pos": pos, "log": log})
    return res


def epochs(j):
    return sum(1 for l in open(j["log"]) if "Mean train" in l)


def measure(running, n):
    e0 = {j["pid"]: epochs(j) for j in running}; t0 = time.time()
    while not all(epochs(j) >= e0[j["pid"]] + n for j in running):
        time.sleep(5)
    dt = time.time() - t0
    per = {f"s{j['seed']}_{j['pos']}": (epochs(j) - e0[j["pid"]]) / dt * 3600 for j in running}
    return {"per_job_epochs_per_h": per, "total_epochs_per_h": sum(per.values()), "window_s": dt}


def stop(js, sig):
    for j in js:
        for p in [j["pid"]] + j["kids"]:
            try:
                os.kill(p, sig)
            except ProcessLookupError:
                pass


ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=2); a = ap.parse_args()
J = sorted(jobs(), key=lambda j: (j["seed"], j["pos"]))
assert len(J) == 4, [(j["seed"], j["pos"]) for j in J]
r = {"4": measure(J, a.epochs)}
stop(J[2:], signal.SIGSTOP)
try:
    r["2"] = measure(J[:2], a.epochs)
    stop(J[1:2], signal.SIGSTOP)
    r["1"] = measure(J[:1], a.epochs)
finally:
    stop(J, signal.SIGCONT)
r["best"] = max(("4", "2", "1"), key=lambda k: r[k]["total_epochs_per_h"])
json.dump(r, open("/workspace/logs/p1_loc_throughput.json", "w"), indent=1); print(json.dumps(r, indent=1))
