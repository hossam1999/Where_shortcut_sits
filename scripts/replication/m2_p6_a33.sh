#!/bin/bash
# A33: after P8's runs (gate), set the pre-A33 P6 bias-1 runs aside and rerun ERM + EPVT at bias 1 once with the fixed step count.
D=${WTSS_DATA:-/workspace/data}; Q=/root/Where_shortcut_sits/scripts/replication
until grep -q "P8 step-1 runs done" /workspace/logs/m2_p8_queue.log 2>/dev/null; do sleep 300; done
for m in erm epvt; do [ -d $D/epvt/runs/${m}_bias1_1 ] && mv $D/epvt/runs/${m}_bias1_1 $D/epvt/runs/preA33_${m}_bias1_1; done
P6_START=after BIASES=1 $Q/m2_p6_queue.sh
