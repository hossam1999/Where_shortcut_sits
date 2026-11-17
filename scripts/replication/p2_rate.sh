#!/bin/bash
# P2 training throughput (images/s) over the last N "Batch" log lines of a run (batch 12; their trainer logs every 168).
L=${1:-/workspace/data/replication/pahde/runs/vanilla_resnet50d_seed0/train.log}; N=${2:-6}
grep "Batch" $L | tail -$N | python3 -c "
import sys,datetime as dt
r=[l.split() for l in sys.stdin]; t=[dt.datetime.fromisoformat(x[0]+' '+x[1].split(',')[0]) for x in r]; b=[int(x[-1].split('/')[0]) for x in r]
pairs=[(b[i]-b[i-1],(t[i]-t[i-1]).total_seconds()) for i in range(1,len(b)) if b[i]>b[i-1]]
n=sum(p[0] for p in pairs); s=sum(p[1] for p in pairs); print(f'{12*n/s:.1f} img/s over {s:.0f}s' if s else 'n/a')"
