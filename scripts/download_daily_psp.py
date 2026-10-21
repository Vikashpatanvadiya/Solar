#!/usr/bin/env python3
"""Download the Grid-India daily PSP Excel files listed in 'PSP Reports/daily/manifest.txt'.

Only the exact manifest entries are fetched, only .xls files, only from the Grid-India
file host. Files already on disk are skipped, so it is safe to re-run.
File names are stored as <YYYY-MM-DD>__<original name>.xls under PSP Reports/daily/files/.
"""
import re
import sys
import time
import subprocess
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DAILY_DIR = ROOT / "PSP Reports" / "daily"
MANIFEST = DAILY_DIR / "manifest.txt"
OUT_DIR = DAILY_DIR / "files"
BASE = "https://webcdn.grid-india.in/files/grdw/"


def local_name(rel: str) -> str:
    name = urllib.parse.unquote(rel.rsplit("/", 1)[-1])
    m = re.match(r"^(\d\d)\.(\d\d)\.(\d\d)_", name)
    if not m:
        raise ValueError(f"unexpected file name: {name}")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return f"20{m.group(3)}-{m.group(2)}-{m.group(1)}__{safe}"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    entries = [l.strip() for l in MANIFEST.read_text().splitlines() if l.strip()]
    ok = skipped = failed = 0
    for rel in entries:
        if not rel.lower().endswith(".xls"):
            raise ValueError(f"manifest entry is not an .xls file: {rel}")
        dest = OUT_DIR / local_name(rel)
        if dest.exists() and dest.stat().st_size > 0:
            skipped += 1
            continue
        url = BASE + urllib.parse.quote(urllib.parse.unquote(rel), safe="/&")
        try:
            # curl is used (TLS verification stays on) because Python's bundled CA store
            # cannot validate this host's certificate chain.
            data = subprocess.run(
                ["curl", "-fsSL", "--max-time", "60", url], capture_output=True, check=True
            ).stdout
            if not data.startswith(b"\xd0\xcf\x11\xe0"):  # OLE2 header of a genuine .xls
                raise ValueError("response is not an .xls workbook")
            dest.write_bytes(data)
            ok += 1
        except Exception as e:  # keep going; report at the end
            failed += 1
            print(f"FAILED {rel}: {e}", file=sys.stderr)
        time.sleep(0.25)
    print(f"downloaded {ok}, already present {skipped}, failed {failed}, manifest {len(entries)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
