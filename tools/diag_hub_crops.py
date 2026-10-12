"""Crop the merge HUBS: are they text, artwork, or screentone? Side by side."""
import glob, json, sys
from pathlib import Path
import cv2, numpy as np
sys.stdout.reconfigure(encoding="utf-8")

MIN_AREA, MAX_AREA, D = 50, 12000, 8
HUBS = {
    "ARMS/000": [(1204,795,158,167),(1390,813,145,169),(1075,942,187,113)],
    "ARMS/004": [(1070,429,476,119),(1327,281,163,204)],
    "ARMS/005": [(1440,654,179,206),(909,393,154,154),(864,219,169,170)],
}
out = Path("deliverables/diag/hubs"); out.mkdir(parents=True, exist_ok=True)
PAD = 25; SCALE = 2
for path in sorted(glob.glob("outputs/translation/spec004-vilao/*/*/*/metadata.json")):
    rec = json.loads(Path(path).read_text(encoding="utf-8"))
    if rec["image_id"] not in HUBS: continue
    page = Path(path).parent
    orig = cv2.imread(str(page/"original.png")); inp = cv2.imread(str(page/"inpainted.png"))
    ren = cv2.imread(str(page/"rendered.png"))
    mask = cv2.imread(str(page/"mask.png"), cv2.IMREAD_GRAYSCALE)
    ov = orig.copy(); ov[mask>0] = (0,0,255)
    for i,(x,y,w,h) in enumerate(HUBS[rec["image_id"]]):
        x0,y0=max(0,x-PAD),max(0,y-PAD); x1,y1=min(1654,x+w+PAD),min(1170,y+h+PAD)
        tiles=[]
        for base in (orig, ov, inp, ren):
            c=base[y0:y1,x0:x1].copy()
            cv2.rectangle(c,(x-x0,y-y0),(x-x0+w,y-y0+h),(0,255,0),1)
            tiles.append(cv2.resize(c,None,fx=SCALE,fy=SCALE,interpolation=cv2.INTER_NEAREST))
        hh=min(t.shape[0] for t in tiles); tiles=[t[:hh] for t in tiles]
        cv2.imwrite(str(out/f"{rec['image_id'].replace('/','_')}_hub{i}.png"), np.hstack(tiles))
    # which regions in metadata cover these hubs?
    for r in rec["regions"]:
        rx,ry,rw,rh = r["bbox"]
        for i,(x,y,w,h) in enumerate(HUBS[rec["image_id"]]):
            if rx<=x and ry<=y and rx+rw>=x+w and ry+rh>=y+h:
                print(f"{rec['image_id']} hub{i} ({x},{y},{w},{h}) -> region {r['region_id']} "
                      f"bbox=({rx},{ry},{rw},{rh}) merged={r['merged_from']} "
                      f"ocr={r.get('ocr_text')!r} vi={r.get('translation')!r} "
                      f"render={r.get('render_status')} font={r.get('font_size_used')}")
print(out)
