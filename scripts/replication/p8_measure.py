"""Measure the P8 jobs running now: epochs per job over one common window (ends after WINDOW s, or when any job exits or
reaches 100 epochs), whole-machine CPU busy % over the window (/proc/stat), and throughput = sum(alone s/epoch / measured).
    p8_measure.py <out.json> <n_jobs> [window_s]"""
import json, os, re, subprocess, sys, time, glob
out, n, W = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]) if len(sys.argv) > 3 else 1200
D = os.environ.get("WTSS_DATA", "/workspace/data"); PY = "/workspace/envs/disc/bin/python"
pj = json.load(open("/workspace/logs/pair_p6p8.json")) if os.path.exists("/workspace/logs/pair_p6p8.json") else {}
ALONE = {"erm": 9.6, "disc": pj.get("disc_s_per_epoch_alone", 96.6)}
def jobs():
    r, P = {}, []
    for l in subprocess.run(["ps", "-eo", "pid,ppid,args"], capture_output=True, text=True).stdout.splitlines()[1:]:
        pid, ppid, a = l.strip().split(None, 2)
        if a.startswith(PY + " run_expt.py"): P.append((pid, ppid, a))
    pids = {p for p, _, _ in P}
    for pid, ppid, a in P:
        if ppid not in pids:   # main process only (DataLoader workers share its command line)
            k = re.search(r"--seed (\d+)", a).group(1); m = "disc" if "--disc" in a else "erm"
            f = glob.glob(f"{D}/disc/output/ISIC/{m.upper()}/*-trapset_id={k}/test.csv")
            r[int(pid)] = (m, k, f[0] if f else None)
    return r
rows = lambda f: sum(1 for _ in open(f)) - 1 if f and os.path.exists(f) else -1
cpu = lambda: (lambda v: (sum(v) - v[3] - v[4], sum(v)))(list(map(int, open("/proc/stat").readline().split()[1:])))
while True:   # wait for n jobs, each with >= 2 epochs logged
    J = jobs()
    if len(J) >= n and all(rows(f) >= 2 for _, _, f in J.values()): break
    time.sleep(10)
c0, t0, r0 = cpu(), time.time(), {p: rows(f) for p, (_, _, f) in J.items()}
while time.time() - t0 < W:
    time.sleep(10)
    if set(jobs()) != set(J) or any(rows(f) >= 100 for _, _, f in J.values()): break
c1, t1 = cpu(), time.time()
res = {"jobs": [], "window_s": t1 - t0, "cpu_busy_pct": 100 * (c1[0] - c0[0]) / (c1[1] - c0[1])}
for p, (m, k, f) in J.items():
    e = rows(f) - r0[p]; s = (t1 - t0) / e if e > 0 else float("inf")
    res["jobs"].append({"job": f"{m} {k}", "epochs": e, "s_per_epoch": s, "alone": ALONE[m], "ratio": ALONE[m] / s})
res["total"] = sum(j["ratio"] for j in res["jobs"]); res["valid"] = all(j["epochs"] >= 3 for j in res["jobs"])
json.dump(res, open(out, "w"), indent=1); print(json.dumps(res))
