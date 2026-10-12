"""Is the mask text-only, or does it cover artwork/screentone? Per page."""
import glob, json, sys
from pathlib import Path
import cv2, numpy as np
sys.stdout.reconfigure(encoding="utf-8")
MIN_AREA, MAX_AREA = 50, 12000
PAGE = 1654 * 1170
print(f"{'image_id':30s} {'mask%':>6s} {'cc':>5s} {'artblob':>7s} {'art%pg':>7s} {'artpx':>7s} {'glyph':>6s} {'bigCC%':>7s}")
print("-"*84)
rows=[]
for path in sorted(glob.glob("outputs/translation/spec004-vilao/*/*/*/metadata.json")):
    rec = json.loads(Path(path).read_text(encoding="utf-8"))
    m = cv2.imread(str(Path(path).parent/"mask.png"), cv2.IMREAD_GRAYSCALE)
    _,_,stats,_ = cv2.connectedComponentsWithStats(m, connectivity=8)
    st = stats[1:]
    # "art-like": a component whose bbox is big in BOTH dims. A manga glyph is ~20-60px.
    art = [s for s in st if s[2]>=90 and s[3]>=90]
    artpx = sum(int(s[4]) for s in art)
    glyph = sum(1 for s in st if 20<=s[2]<=70 and 20<=s[3]<=70 and MIN_AREA<=s[4]<=MAX_AREA)
    bigcc = max(int(s[4]) for s in st)/PAGE if len(st) else 0
    rows.append((rec["image_id"], m.mean()/255*100, len(st), len(art), artpx/PAGE*100, artpx, glyph, bigcc*100))
for r in sorted(rows, key=lambda r:-r[4]):
    print(f"{r[0]:30s} {r[1]:5.1f}% {r[2]:5d} {r[3]:7d} {r[4]:6.1f}% {r[5]:7d} {r[6]:6d} {r[7]:6.2f}%")
print("-"*84)
tot_art = sum(r[5] for r in rows); tot_mask = sum(r[1] for r in rows)/100*PAGE*len(rows)
print(f"mask pixels inside art-like blobs: {tot_art} ({100*tot_art/tot_mask:.1f}% of all mask ink)")
