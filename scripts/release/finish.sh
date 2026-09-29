#!/usr/bin/env bash
# Release handoff after Part A: C4 archive, C5 manifest, D1 checks, D3 secret scan, summary, D4 commit/push to both
# branches, D5 credential removal (only after the push is confirmed). Log: results/rerun_2026-09-28/_logs/finish.log
#   bash scripts/release/finish.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
source /venv/main/bin/activate
export PYTHONPATH="$PWD/src" WTSS_DATA=/root/data WTSS_BOOTSTRAP=crossed TQDM_DISABLE=1
RR=results/rerun_2026-09-28; L=$RR/_logs; BR=claude/festive-bohr-95evo5
TRAILER="Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0192Mq7C8i7t15VkZPfVxwbx"
log(){ echo "=== $(date +%T) $*"; }

log "waiting for Part A"
until grep -q "PART A DONE" $L/finish_A.log; do sleep 30; done

# ---- C4 / C5 ---------------------------------------------------------------------------------------------------------
log "C4 archive"
python scripts/verify/build_release_archive.py > $L/C4_archive.log 2>&1; echo "archive rc=$?"
( cd /root/release_archive && sha256sum -c SHA256SUMS ) > $L/C4_verify.log 2>&1; echo "sha256sum -c rc=$?"
ls -la /root/release_archive/ | grep -v staging

# ---- D1 --------------------------------------------------------------------------------------------------------------
log "D1 make test"; make test > $L/D1_make_test.log 2>&1; echo "make test rc=$?"; tail -2 $L/D1_make_test.log
log "D1 audit_numbers_final --docs all"
python scripts/verify/audit_numbers_final.py --docs all > $L/D1_audit_numbers.log 2>&1; echo "audit rc=$?"; tail -4 $L/D1_audit_numbers.log
log "D1 make verify"; make verify > $L/D1_make_verify.log 2>&1; echo "make verify rc=$?"; tail -4 $L/D1_make_verify.log

# ---- D3: secret scan of the full history (report pattern counts, never the matched text) -----------------------------
log "D3 secret scan"
git log -p --all > /tmp/claude-0/histscan.txt 2>/dev/null
python - > $L/D3_secret_scan.log <<'PY'
import re
t = open("/tmp/claude-0/histscan.txt", errors="ignore").read()
pats = {"Kaggle API token (KGAT_...)": r"KGAT_[A-Za-z0-9]{8,}", "Hugging Face token (hf_ + 30 chars)": r"hf_[A-Za-z0-9]{30,}",
        "GitHub token (ghp_/gho_/ghs_/github_pat_)": r"(gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})",
        "kaggle.json key field": r'"key"\s*:\s*"[0-9a-f]{32}"', "private key block": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
        "assignment of a token value (token=/api_key=/password= followed by >= 20 chars)":
            r"(?i)(token|api[_-]?key|password|secret)\s*[=:]\s*['\"]?[A-Za-z0-9_\-]{20,}"}
print(f"history scanned: {len(t)} characters")
for k, p in pats.items():
    print(f"{k}: {len(re.findall(p, t))} match(es)")
print("mentions of the word 'kaggle' (commands and documentation, not credentials):", len(re.findall(r"(?i)kaggle", t)))
PY
rm -f /tmp/claude-0/histscan.txt; cat $L/D3_secret_scan.log

# ---- summary (D2) ------------------------------------------------------------------------------------------------------
log "summary"
python scripts/release/write_summary.py
git add results/ARCHIVE_MANIFEST.csv $RR/SUMMARY_archived_reruns.md results/rerun_2026-09-28/SUMMARY.md \
  scripts/release scripts/verify/build_release_archive.py docs/AFTER_RELEASE.md 2>/dev/null
git commit -q -m "Release handoff: archive manifest (C5), summary of the archived-run regeneration and final checks (D1-D3)

$TRAILER"
# the author's own local change to the rating tool (page layout only; no rating content), committed separately
if ! git diff --quiet -- scripts/round6/rating_app.py; then
  git add scripts/round6/rating_app.py
  git commit -q -m "Rating tool: show the original image beside the overlay (author's local change, committed at handoff)

$TRAILER"
fi

# ---- D4 --------------------------------------------------------------------------------------------------------------
log "D4 push"
git push origin "$BR" && git push origin "$BR:stage-reports"
git fetch -q origin
st=$(git status --porcelain --untracked-files=no)
a=$(git rev-parse HEAD); b=$(git rev-parse origin/$BR); c=$(git rev-parse origin/stage-reports)
echo "HEAD $a | origin/$BR $b | origin/stage-reports $c | tracked changes: ${st:-none}"
echo "untracked (not committed on purpose): $(git status --porcelain | grep '^??' | wc -l) path(s)"
if [ "$a" = "$b" ] && [ "$a" = "$c" ] && [ -z "$st" ]; then PUSHED=1; echo "PUSH CONFIRMED"; else PUSHED=0; echo "PUSH NOT CONFIRMED: credentials kept"; fi

# ---- D5: remove credentials only after the push is confirmed ----------------------------------------------------------
if [ "$PUSHED" = 1 ]; then
  log "D5 remove credentials"
  for f in ~/.kaggle/kaggle.json ~/.git-credentials ~/.cache/huggingface/token /workspace/.hf_home/token \
           /workspace/.hf_home/stored_tokens ~/.config/gh/hosts.yml ~/.netrc ~/.ssh/id_ed25519 ~/.ssh/id_ed25519.pub; do
    [ -e "$f" ] && rm -f "$f" && echo "removed $f"
  done
  sed -i -E '/^(HF_TOKEN|KAGGLE_API_TOKEN|KAGGLE_KEY|KAGGLE_USERNAME|GH_TOKEN|GITHUB_TOKEN)=/d' /workspace/.env && echo "removed token lines from /workspace/.env"
  # any text file still containing a Kaggle or HF token string (shell history, snapshots, logs)
  grep -rlI -E "KGAT_[A-Za-z0-9]{8,}|hf_[A-Za-z0-9]{30,}" /root /workspace --exclude-dir=release_archive --exclude-dir=.claude 2>/dev/null \
    | grep -v -E "/proc/" | while read -r f; do sed -i -E 's/KGAT_[A-Za-z0-9]{8,}/KGAT_REMOVED/g; s/hf_[A-Za-z0-9]{30,}/hf_REMOVED/g' "$f" && echo "scrubbed $f"; done
  echo "not scrubbed: /root/.claude (this session's own transcript, which contains the token pasted in chat; it goes with the machine)"
  echo "remaining credential files: $(ls ~/.kaggle/kaggle.json ~/.git-credentials ~/.cache/huggingface/token /workspace/.hf_home/token ~/.ssh/id_ed25519 2>/dev/null | wc -l)"
fi
log "FINISH DONE"
