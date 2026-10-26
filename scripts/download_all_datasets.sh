#!/usr/bin/env bash
# =============================================================================
# download_all_datasets.sh  (macOS; bash 3.2 compatible)
#
# Downloads the five project datasets into ./data/raw (or --root DIR):
#   1. paderborn  Paderborn KAt Bearing DataCenter   32 .rar (~5.1 GB)   university server
#   2. xjtu       XJTU-SY run-to-failure              Google Drive folder via gdown
#   3. femto      FEMTO / PRONOSTIA                   NASA PCoE mirror zip
#   4. cmapss     NASA C-MAPSS FD001-FD004            NASA PCoE mirror zip
#   5. cwru       CWRU Bearing Data Center            crawled .mat links
#
# Design:
#   - Each dataset runs independently: a failure is logged and the script continues.
#   - Downloads resume (curl -C -); complete files are skipped; re-running is safe.
#   - Original archives are never deleted (raw data is read-only).
#   - Writes per-dataset MANIFEST.csv (file, bytes, md5), a log, and SUMMARY.csv.
#
# Usage:
#   bash download_all_datasets.sh
#   bash download_all_datasets.sh --only paderborn,femto
#   bash download_all_datasets.sh --root /Volumes/External/pdm/data/raw --skip-hashes
#   bash download_all_datasets.sh --skip-extract
# =============================================================================

# ----------------------------- options ---------------------------------------
ROOT="./data/raw"
ONLY="paderborn,xjtu,femto,cmapss,cwru"
SKIP_EXTRACT=0
SKIP_HASHES=0
AUTO_INSTALL=1

while [ $# -gt 0 ]; do
  case "$1" in
    --root)          ROOT="$2"; shift 2 ;;
    --only)          ONLY="$2"; shift 2 ;;
    --skip-extract)  SKIP_EXTRACT=1; shift ;;
    --skip-hashes)   SKIP_HASHES=1; shift ;;
    --no-install)    AUTO_INSTALL=0; shift ;;
    -h|--help)       sed -n '2,26p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1"; exit 2 ;;
  esac
done

# Optional: your own consolidated Kaggle datasets (leave empty to use original sources)
KAGGLE_SLUG_PADERBORN=""   # e.g. "yourname/paderborn-kat-consolidated"
KAGGLE_SLUG_XJTU=""
KAGGLE_SLUG_FEMTO=""
KAGGLE_SLUG_CMAPSS=""
KAGGLE_SLUG_CWRU=""

# ----------------------------- sources ---------------------------------------
# FEMTO and C-MAPSS URLs are listed on https://data.phmsociety.org/nasa/
PADERBORN_BASE="https://groups.uni-paderborn.de/kat/BearingDataCenter/"
PADERBORN_PAPER="http://mb.uni-paderborn.de/fileadmin-mb/kat/PDF/Veroeffentlichungen/20160703_PHME16_CM_bearing.pdf"
PADERBORN_CODES="K001 K002 K003 K004 K005 K006 KA01 KA03 KA04 KA05 KA06 KA07 KA08 KA09 KA15 KA16 KA22 KA30 KB23 KB24 KB27 KI01 KI03 KI04 KI05 KI07 KI08 KI14 KI16 KI17 KI18 KI21"
XJTU_DRIVE="https://drive.google.com/drive/folders/1_ycmG46PARiykt82ShfnFfyQsaXv3_VK"
FEMTO_URL="https://phm-datasets.s3.amazonaws.com/NASA/10.+FEMTO+Bearing.zip"
CMAPSS_URL="https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
CWRU_START="https://engineering.case.edu/bearingdatacenter/download-data-file"
UA="Academic dataset download (student project, sequential requests)"

# ----------------------------- setup -----------------------------------------
set -o pipefail
mkdir -p "$ROOT/logs"
ROOT="$(cd "$ROOT" && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$ROOT/logs/download_$STAMP.log"
exec > >(tee -a "$LOG") 2>&1

SUMMARY="$ROOT/SUMMARY.csv"
echo "dataset,status,size_gb,path,notes" > "$SUMMARY"

c_cyan=$'\033[36m'; c_green=$'\033[32m'; c_yellow=$'\033[33m'; c_red=$'\033[31m'; c_off=$'\033[0m'
step() { printf '\n%s=== %s ===%s\n' "$c_cyan" "$1" "$c_off"; }
ok()   { printf '  %s[ok]%s   %s\n' "$c_green" "$c_off" "$1"; }
warn() { printf '  %s[warn]%s %s\n' "$c_yellow" "$c_off" "$1"; }
fail() { printf '  %s[fail]%s %s\n' "$c_red" "$c_off" "$1"; }

wants() { case ",$ONLY," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }

add_result() {  # dataset status path notes
  local size_gb="0"
  if [ -d "$3" ]; then
    size_gb=$(du -sk "$3" 2>/dev/null | awk '{printf "%.2f", $1/1024/1024}')
  fi
  local notes="${4//,/;}"
  echo "$1,$2,$size_gb,$3,$notes" >> "$SUMMARY"
}

# ----------------------------- tools -----------------------------------------
step "Checking tools"
command -v curl >/dev/null && ok "curl found" || fail "curl missing (required)"

HAS_BREW=0; command -v brew >/dev/null && HAS_BREW=1

# Python venv for gdown + kaggle (avoids 'externally managed environment' errors)
PY=""
if command -v python3 >/dev/null; then
  VENV="$HOME/.cache/pdm_download_venv"      # kept out of data/raw on purpose
  mkdir -p "$HOME/.cache"
  if [ ! -x "$VENV/bin/python" ]; then
    python3 -m venv "$VENV" && ok "created venv $VENV" || warn "could not create venv"
  fi
  if [ -x "$VENV/bin/python" ]; then
    PY="$VENV/bin/python"
    "$PY" -m pip install -q -U pip gdown kaggle >/dev/null 2>&1 && ok "gdown + kaggle installed in venv" \
      || warn "pip install of gdown/kaggle failed"
  fi
else
  warn "python3 not found: XJTU-SY (gdown) and Kaggle downloads will be skipped"
fi

# Archive extractor: 7zz (sevenzip) > 7z > unar > bsdtar (macOS tar)
EXTRACTOR=""
pick_extractor() {
  for t in 7zz 7z unar; do
    if command -v "$t" >/dev/null; then EXTRACTOR="$t"; return; fi
  done
}
pick_extractor
if [ -z "$EXTRACTOR" ] && [ "$AUTO_INSTALL" = "1" ] && [ "$HAS_BREW" = "1" ]; then
  warn "no 7-Zip/unar found, installing 'sevenzip' with Homebrew..."
  brew install sevenzip >/dev/null 2>&1 || brew install unar >/dev/null 2>&1
  pick_extractor
fi
[ -z "$EXTRACTOR" ] && EXTRACTOR="tar"
ok "extractor: $EXTRACTOR"

df -g "$ROOT" | awk 'NR==2 {printf "  free space: %s GB (recommend 40+ GB for everything extracted)\n", $4}'

# ----------------------------- helpers ---------------------------------------
remote_size() {
  curl -sIL -A "$UA" --connect-timeout 30 "$1" 2>/dev/null \
    | tr -d '\r' | grep -i '^content-length:' | tail -1 | awk '{print $2}'
}

local_size() { stat -f%z "$1" 2>/dev/null || echo 0; }

download() {  # url outfile [retries]
  local url="$1" out="$2" retries="${3:-5}" i code remote
  mkdir -p "$(dirname "$out")"
  remote="$(remote_size "$url")"
  if [ -f "$out" ] && [ -n "$remote" ] && [ "$(local_size "$out")" = "$remote" ]; then
    ok "already complete: $(basename "$out")"; return 0
  fi
  for i in $(seq 1 "$retries"); do
    curl -L -f -sS -C - --retry 3 --retry-delay 5 --connect-timeout 30 -A "$UA" -o "$out" "$url"
    code=$?
    if [ "$code" = "33" ]; then           # server refused resume: restart cleanly
      rm -f "$out"
      curl -L -f -sS --retry 3 -A "$UA" -o "$out" "$url"; code=$?
    fi
    if [ "$code" = "0" ]; then
      if [ -n "$remote" ] && [ "$(local_size "$out")" != "$remote" ]; then
        warn "size mismatch on $(basename "$out") ($(local_size "$out") vs $remote)"
      else
        ok "downloaded $(basename "$out") ($(( $(local_size "$out") / 1048576 )) MB)"; return 0
      fi
    else
      warn "attempt $i/$retries for $(basename "$out"): curl exit $code"
    fi
    sleep $(( 10 * i ))
  done
  fail "giving up: $url"
  return 1
}

extract() {  # archive dest
  local a="$1" d="$2"
  [ "$SKIP_EXTRACT" = "1" ] && return 0
  if [ -d "$d" ] && [ -n "$(find "$d" -type f -print -quit 2>/dev/null)" ]; then
    return 0                               # already extracted earlier
  fi
  mkdir -p "$d"
  case "$EXTRACTOR" in
    7zz|7z) "$EXTRACTOR" x "$a" "-o$d" -y >/dev/null ;;
    unar)   unar -q -f -o "$d" "$a" >/dev/null ;;
    tar)
      case "$a" in
        *.zip) ditto -x -k "$a" "$d" ;;
        *)     tar -xf "$a" -C "$d" ;;
      esac ;;
  esac
  local rc=$?
  if [ $rc -gt 1 ]; then warn "extract exit $rc on $(basename "$a")"; return 1; fi
  return 0
}

is_later_rar_part() { echo "$1" | grep -Eiq '\.part0*([2-9]|[0-9]{2,})\.rar$'; }

extract_nested() {  # dir : extract archives found inside an extracted tree
  [ "$SKIP_EXTRACT" = "1" ] && return 0
  local dir="$1" done_list pass f
  done_list="$(mktemp)"
  for pass in 1 2 3 4 5; do
    local found=0
    while IFS= read -r f; do
      grep -Fxq "$f" "$done_list" && continue
      is_later_rar_part "$f" && continue
      echo "$f" >> "$done_list"; found=1
      echo "  nested archive: $(basename "$f")"
      extract "$f" "${f%.*}"
    done < <(find "$dir" -type f \( -iname '*.zip' -o -iname '*.rar' -o -iname '*.7z' \) 2>/dev/null)
    [ "$found" = "0" ] && break
  done
  rm -f "$done_list"
}

write_manifest() {  # dir out_csv
  [ "$SKIP_HASHES" = "1" ] && return 0
  [ -d "$1" ] || return 0
  echo "file,bytes,md5" > "$2"
  find "$1" -type f \( -iname '*.zip' -o -iname '*.rar' -o -iname '*.7z' -o -iname '*.mat' \) | sort | \
  while IFS= read -r f; do
    echo "${f#$1/},$(local_size "$f"),$(md5 -q "$f")" >> "$2"
  done
}

from_kaggle() {  # slug dest -> 0 if downloaded
  [ -z "$1" ] && return 1
  [ -z "$PY" ] && { warn "no Python/Kaggle CLI, using original source"; return 1; }
  echo "  trying your Kaggle dataset $1 ..."
  mkdir -p "$2"
  if "$PY" -m kaggle datasets download -d "$1" -p "$2" --unzip; then ok "Kaggle copy downloaded"; return 0; fi
  warn "Kaggle download failed, using original source"; return 1
}

# ============================================================================
# 1. PADERBORN
# ============================================================================
do_paderborn() {
  local base="$ROOT/paderborn" arch ext docs code failed="" counts="" n status
  if from_kaggle "$KAGGLE_SLUG_PADERBORN" "$base/kaggle_consolidated"; then
    add_result paderborn "OK (Kaggle copy)" "$base" "consolidated HDF5 + CSVs"; return 0
  fi
  arch="$base/archives"; ext="$base/extracted"; docs="$base/docs"
  download "${PADERBORN_BASE}readme_versions.txt" "$docs/readme_versions.txt" 3
  download "$PADERBORN_PAPER" "$docs/PHME2016_Lessmeier.pdf" 3
  for code in $PADERBORN_CODES; do
    if ! download "$PADERBORN_BASE$code.rar" "$arch/$code.rar"; then failed="$failed $code"; continue; fi
    if extract "$arch/$code.rar" "$ext/$code" && [ "$SKIP_EXTRACT" = "0" ]; then
      n=$(find "$ext/$code" -type f -iname '*.mat' | wc -l | tr -d ' ')
      if [ "$n" != "80" ]; then counts="$counts $code=$n"; warn "$code has $n .mat files (expected 80)"; fi
    fi
    sleep 2
  done
  write_manifest "$arch" "$base/MANIFEST.csv"
  if [ -z "$failed" ]; then status="OK"; else status="PARTIAL"; fi
  add_result paderborn "$status" "$base" "failed:${failed:- none} | count issues:${counts:- none}"
}

# ============================================================================
# 2. XJTU-SY
# ============================================================================
do_xjtu() {
  local base="$ROOT/xjtu" dl ext f n status
  if from_kaggle "$KAGGLE_SLUG_XJTU" "$base/kaggle_consolidated"; then
    add_result xjtu "OK (Kaggle copy)" "$base" "consolidated HDF5 + CSVs"; return 0
  fi
  dl="$base/download"; ext="$base/extracted"; mkdir -p "$dl"
  if [ -z "$(find "$dl" -type f -print -quit)" ]; then
    if [ -z "$PY" ]; then fail "Python needed for gdown"; add_result xjtu FAILED "$base" "no python"; return 1; fi
    "$PY" -m gdown --folder "$XJTU_DRIVE" -O "$dl" --remaining-ok || warn "gdown returned an error"
  else
    ok "files already in $dl (re-using them)"
  fi
  if [ -z "$(find "$dl" -type f -print -quit)" ]; then
    fail "nothing downloaded (Google Drive may be rate-limiting)"
    echo "  MANUAL STEP: open $XJTU_DRIVE in a browser, click Download,"
    echo "  move the zip file(s) into $dl, then run:  bash $0 --only xjtu"
    add_result xjtu MANUAL_STEP_NEEDED "$base" "gdown failed; manual download"; return 1
  fi
  while IFS= read -r f; do
    is_later_rar_part "$f" && continue
    extract "$f" "$ext/$(basename "${f%.*}")"
  done < <(find "$dl" -type f \( -iname '*.zip' -o -iname '*.rar' -o -iname '*.7z' \))
  [ -d "$ext" ] && extract_nested "$ext"
  n=$(find "$dl" "$ext" -type d -name 'Bearing*_*' 2>/dev/null | while IFS= read -r d; do
        ls "$d"/*.csv >/dev/null 2>&1 && basename "$d"; done | sort -u | wc -l | tr -d ' ')
  if [ "$n" = "15" ]; then ok "found 15 bearing folders"; status="OK"
  else warn "found $n bearing folders (expected 15)"; status="PARTIAL"; fi
  write_manifest "$dl" "$base/MANIFEST.csv"
  add_result xjtu "$status" "$base" "bearing folders: $n"
}

# ============================================================================
# 3. FEMTO
# ============================================================================
do_femto() {
  local base="$ROOT/femto" zip ext nb sets full notemp
  if from_kaggle "$KAGGLE_SLUG_FEMTO" "$base/kaggle_consolidated"; then
    add_result femto "OK (Kaggle copy)" "$base" "consolidated"; return 0
  fi
  zip="$base/archives/FEMTO_Bearing.zip"; ext="$base/extracted"
  download "$FEMTO_URL" "$zip" || { add_result femto FAILED "$base" "download failed"; return 1; }
  extract "$zip" "$ext"; extract_nested "$ext"
  nb=$(find "$ext" -type d -name 'Bearing*_*' | wc -l | tr -d ' ')
  sets=$(find "$ext" -type d -name 'Bearing*_*' -exec dirname {} \; | xargs -I{} basename {} | sort -u | tr '\n' '/')
  full=$(echo "$sets" | grep -io 'full' | head -1)
  echo "  bearing folders: $nb | sets: $sets"
  [ -z "$full" ] && warn "no Full_Test_Set folder: test bearings may be truncated before failure"
  notemp=$(find "$ext" -type d -name 'Bearing*_*' | while IFS= read -r d; do
             ls "$d"/temp_*.csv >/dev/null 2>&1 || echo "$(basename "$(dirname "$d")")/$(basename "$d")"; done)
  [ -n "$notemp" ] && warn "no temperature files in: $(echo $notemp)"
  write_manifest "$base/archives" "$base/MANIFEST.csv"
  add_result femto OK "$base" "folders: $nb; sets: $sets; full test set: ${full:-no}; missing temp: $(echo "$notemp" | grep -c . )"
}

# ============================================================================
# 4. C-MAPSS
# ============================================================================
do_cmapss() {
  local base="$ROOT/cmapss" zip ext n status
  if from_kaggle "$KAGGLE_SLUG_CMAPSS" "$base/kaggle_consolidated"; then
    add_result cmapss "OK (Kaggle copy)" "$base" "consolidated"; return 0
  fi
  zip="$base/archives/CMAPSS.zip"; ext="$base/extracted"
  download "$CMAPSS_URL" "$zip" || { add_result cmapss FAILED "$base" "download failed"; return 1; }
  extract "$zip" "$ext"; extract_nested "$ext"
  n=$(find "$ext" -type f -name 'train_FD00*.txt' | wc -l | tr -d ' ')
  if [ "$n" = "4" ]; then ok "train_FD001-FD004 present"; status="OK"
  else warn "found $n train files (expected 4)"; status="PARTIAL"; fi
  write_manifest "$base/archives" "$base/MANIFEST.csv"
  add_result cmapss "$status" "$base" "train files: $n"
}

# ============================================================================
# 5. CWRU (crawl official pages; Python stdlib only)
# ============================================================================
do_cwru() {
  local base="$ROOT/cwru" mat total failed=0 file url status
  if from_kaggle "$KAGGLE_SLUG_CWRU" "$base/kaggle_consolidated"; then
    add_result cwru "OK (Kaggle copy)" "$base" "consolidated"; return 0
  fi
  mat="$base/raw_mat"; mkdir -p "$mat"
  local pybin="${PY:-python3}"
  if ! command -v "$pybin" >/dev/null && [ ! -x "$pybin" ]; then
    fail "python3 needed for crawling"; add_result cwru FAILED "$base" "no python"; return 1
  fi
  "$pybin" - "$CWRU_START" "$base" "$UA" <<'PYEOF'
import csv, re, sys, urllib.parse, urllib.request
start, base, ua = sys.argv[1], sys.argv[2], sys.argv[3]
def get(u):
    req = urllib.request.Request(u, headers={"User-Agent": ua})
    return urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "replace")
strip = lambda h: re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", h)).strip()
html = get(start)
pages = {start}
for href in re.findall(r'href="([^"]+)"', html, re.I):
    u = urllib.parse.urljoin(start, href).split("#")[0]
    if "bearingdatacenter" in u and not u.lower().endswith(".mat"):
        pages.add(u)
rows, seen = [], set()
for p in sorted(pages):
    try:
        h = get(p)
    except Exception:
        continue
    m = re.search(r"<h1[^>]*>(.*?)</h1>", h, re.I | re.S)
    title = strip(m.group(1)) if m else ""
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", h, re.I | re.S):
        for href in re.findall(r'href="([^"]+\.mat)"', tr, re.I):
            u = urllib.parse.urljoin(p, href)
            if u in seen: continue
            seen.add(u)
            rows.append([title, p, strip(tr), u.rsplit("/", 1)[-1], u])
    for href in re.findall(r'href="([^"]+\.mat)"', h, re.I):
        u = urllib.parse.urljoin(p, href)
        if u in seen: continue
        seen.add(u)
        rows.append([title, p, "", u.rsplit("/", 1)[-1], u])
with open(f"{base}/catalogue.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["page_title", "page_url", "row_text", "file", "url"]); w.writerows(rows)
with open(f"{base}/urls.tsv", "w", encoding="utf-8") as f:
    for r in rows: f.write(f"{r[3]}\t{r[4]}\n")
print(f"  pages scanned: {len(pages)} | .mat links found: {len(rows)}")
PYEOF
  if [ ! -s "$base/urls.tsv" ]; then
    fail "no .mat links found (page layout changed or access blocked)"
    echo "  MANUAL STEP: download the .mat files from $CWRU_START into $mat"
    add_result cwru FAILED "$base" "crawl found no links"; return 1
  fi
  total=$(wc -l < "$base/urls.tsv" | tr -d ' ')
  while IFS=$'\t' read -r file url; do
    download "$url" "$mat/$file" 3 || failed=$((failed + 1))
    sleep 0.5
  done < "$base/urls.tsv"
  write_manifest "$mat" "$base/MANIFEST.csv"
  if [ "$failed" = "0" ]; then status="OK"; else status="PARTIAL"; fi
  add_result cwru "$status" "$base" "files: $total; failed: $failed; labels in catalogue.csv row_text"
}

# ----------------------------- run all ---------------------------------------
run() {  # name function: never lets one failure stop the script
  step "$1"
  "$2" || warn "$1 finished with problems (see above); continuing"
}
wants paderborn && run "1/5 Paderborn KAt"        do_paderborn
wants xjtu      && run "2/5 XJTU-SY"              do_xjtu
wants femto     && run "3/5 FEMTO / PRONOSTIA"    do_femto
wants cmapss    && run "4/5 NASA C-MAPSS"         do_cmapss
wants cwru      && run "5/5 CWRU"                 do_cwru

step "Summary"
column -s, -t < "$SUMMARY"
echo
echo "Summary: $SUMMARY"
echo "Log:     $LOG"
echo "Re-run any time: complete files are skipped, partial downloads resume."
exit 0
