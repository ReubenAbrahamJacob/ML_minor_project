"""Download datasets into data/raw/<key>/ and write data/raw/MANIFEST.md.

Usage: python scripts/download_data.py [cwru femto cmapss xjtu]
CWRU: crawled from the official site (12k drive-end faults + normal baseline).
FEMTO / C-MAPSS: NASA PHM mirror zips. XJTU-SY: Google Drive folder via gdown.
Resumable; complete files are skipped. Raw data is never deleted.
"""
import re
import subprocess
import sys
import zipfile
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
UA = {"User-Agent": "Mozilla/5.0 (academic student project)"}
CWRU_PAGES = [
    "https://engineering.case.edu/bearingdatacenter/normal-baseline-data",
    "https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data",
]
ZIPS = {
    "femto": "https://phm-datasets.s3.amazonaws.com/NASA/10.+FEMTO+Bearing.zip",
    "cmapss": "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip",
}
XJTU_DRIVE = "https://drive.google.com/drive/folders/1_ycmG46PARiykt82ShfnFfyQsaXv3_VK"


def fetch(url: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 0:
        return
    subprocess.run(["curl", "-L", "-f", "-sS", "-C", "-", "--retry", "5", "-A", UA["User-Agent"],
                    "-o", str(out), url], check=True)


def cwru() -> None:
    d = RAW / "cwru"
    lines = []
    for page in CWRU_PAGES:
        html = subprocess.run(["curl", "-sSL", "-A", UA["User-Agent"], page], capture_output=True, text=True, check=True).stdout
        for url, label in re.findall(r'<a href="([^"]+\.mat)">([^<]+)</a>', html):
            lines.append((url.rsplit("/", 1)[-1], label.strip(), url))
    # normal-baseline links are plain numbers (97.mat..100.mat); label them Normal_<load>
    rows = []
    for f, label, url in sorted(set(lines)):
        if f in {"97.mat", "98.mat", "99.mat", "100.mat"}:
            label = f"Normal_{int(f[:-4]) - 97}"
        fetch(url, d / "raw_mat" / f)
        rows.append(f"{f},{label}")
    (d / "catalogue.csv").write_text("file,label\n" + "\n".join(rows) + "\n")


def zipped(key: str) -> None:
    d = RAW / key
    z = d / {"femto": "FEMTO_Bearing.zip", "cmapss": "CMAPSS.zip"}[key]
    fetch(ZIPS[key], z)
    if not (d / "extracted").exists():
        with zipfile.ZipFile(z) as zf:
            zf.extractall(d / "extracted")
    # NASA zips contain nested zips: unpack until none are left
    for _ in range(4):
        inner = [p for p in (d / "extracted").rglob("*.zip") if not p.with_suffix("").exists()]
        for p in inner:
            with zipfile.ZipFile(p) as zf:
                zf.extractall(p.with_suffix(""))


def xjtu() -> None:
    d = RAW / "xjtu" / "download"
    d.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "gdown", "--folder", XJTU_DRIVE, "-O", str(d)],
                   check=False)


def manifest() -> None:
    out = ["# Raw data manifest\n", "| dataset | files | size (MB) | status |", "|---|---|---|---|"]
    for key in ["cwru", "femto", "xjtu", "cmapss"]:
        files = [p for p in (RAW / key).rglob("*") if p.is_file() and p.suffix != ".log"] if (RAW / key).exists() else []
        mb = sum(p.stat().st_size for p in files) / 1e6
        out.append(f"| {key} | {len(files)} | {mb:.0f} | {'present' if files else 'MISSING'} |")
    (RAW / "MANIFEST.md").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    todo = sys.argv[1:] or ["cwru", "femto", "cmapss", "xjtu"]
    for k in todo:
        try:
            {"cwru": cwru, "femto": lambda: zipped("femto"), "cmapss": lambda: zipped("cmapss"), "xjtu": xjtu}[k]()
            print("done", k)
        except Exception as e:  # keep going; report at the end
            print("FAILED", k, e)
    manifest()
