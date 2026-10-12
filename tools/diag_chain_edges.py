"""Trace the ACTUAL merge chain of the largest group: which pairs merged, and how.

We rebuild the union-find, keep the spanning-tree edges, and print the edges that
link the biggest group. That names the real "bus" — no more guessing.
"""
import glob, json, sys
from pathlib import Path
import cv2, numpy as np
sys.stdout.reconfigure(encoding="utf-8")

MIN_AREA, MAX_AREA, D = 50, 12000, 8
PAGE = 1654 * 1170

def find(p, i):
    while p[i] != i:
        p[i] = p[p[i]]; i = p[i]
    return i

for path in sorted(glob.glob("outputs/translation/spec004-vilao/*/*/*/metadata.json")):
    rec = json.loads(Path(path).read_text(encoding="utf-8"))
    mask = cv2.imread(str(Path(path).parent / "mask.png"), cv2.IMREAD_GRAYSCALE)
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = [(int(s[0]), int(s[1]), int(s[2]), int(s[3])) for s in stats[1:] if MIN_AREA <= s[4] <= MAX_AREA]
    areas = [int(s[4]) for s in stats[1:] if MIN_AREA <= s[4] <= MAX_AREA]
    n = len(boxes)
    p = list(range(n)); edges = []
    for a in range(n):
        ax, ay, aw, ah = boxes[a]
        for b in range(a + 1, n):
            bx, by, bw, bh = boxes[b]
            hg = max(ax - (bx + bw), bx - (ax + aw), 0)
            vg = max(ay - (by + bh), by - (ay + ah), 0)
            if hg < D and vg < D and find(p, a) != find(p, b):
                edges.append((a, b, hg, vg))
                p[max(find(p,a), find(p,b))] = min(find(p,a), find(p,b))
    groups = {}
    for i in range(n):
        groups.setdefault(find(p, i), []).append(i)
    biggest = max(groups.values(), key=lambda g: (max(boxes[i][0]+boxes[i][2] for i in g)-min(boxes[i][0] for i in g))*(max(boxes[i][1]+boxes[i][3] for i in g)-min(boxes[i][1] for i in g)))
    x0 = min(boxes[i][0] for i in biggest); y0 = min(boxes[i][1] for i in biggest)
    x1 = max(boxes[i][0]+boxes[i][2] for i in biggest); y1 = max(boxes[i][1]+boxes[i][3] for i in biggest)
    share = (x1-x0)*(y1-y0)/PAGE
    print(f"\n=== {rec['image_id']}  biggest group: {len(biggest)} comps, box {x1-x0}x{y1-y0} = {share*100:.1f}% page ===")
    if share < 0.05: continue
    S = set(biggest)
    sub = [(a,b,hg,vg) for (a,b,hg,vg) in edges if a in S and b in S]
    print(f"  spanning edges inside the group: {len(sub)}  (n-1={len(biggest)-1})")
    # which components have the most links (the hubs)?
    from collections import Counter
    deg = Counter()
    for a,b,_,_ in sub: deg[a]+=1; deg[b]+=1
    print("  top hubs (degree, bbox, area, aspect, fill):")
    for i,_ in deg.most_common(6):
        x,y,w,h = boxes[i]; f = areas[i]/max(1,w*h)
        print(f"    deg={deg[i]:3d} ({x:4d},{y:4d},{w:4d},{h:4d}) area={areas[i]:6d} aspect={max(w,h)/max(1,min(w,h)):6.1f} fill={f:.2f}")
    # gap distribution of the linking edges
    hs = [hg for _,_,hg,_ in sub]; vs = [vg for _,_,_,vg in sub]
    print(f"  linking gaps: h median {np.median(hs):.0f} max {max(hs)}   v median {np.median(vs):.0f} max {max(vs)}")
    print(f"  edges that are diagonal (hg>0 and vg>0): {sum(1 for _,_,hg,vg in sub if hg>0 and vg>0)}/{len(sub)}")
