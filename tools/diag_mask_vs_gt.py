"""Precision/recall of the produced mask against the Spec-2 ground truth."""
import glob, json, sys
from pathlib import Path
import cv2, numpy as np
sys.stdout.reconfigure(encoding="utf-8")
GT = Path("data/groundtruth/post-processed")
print(f"{'image_id':28s} {'mask%':>6s} {'gt%':>6s} {'prec':>6s} {'recall':>7s} {'IoU':>6s} {'FPpx':>8s} {'FNpx':>8s}")
print("-"*82)
P,R,IOU=[],[],[]
for path in sorted(glob.glob("outputs/translation/spec004-vilao/*/*/*/metadata.json")):
    rec = json.loads(Path(path).read_text(encoding="utf-8"))
    iid = rec["image_id"]; book, page = iid.split("/")
    gtp = GT/book/f"{page}.png"
    if not gtp.exists(): continue
    m = cv2.imread(str(Path(path).parent/"mask.png"), cv2.IMREAD_GRAYSCALE)
    g = cv2.imread(str(gtp), cv2.IMREAD_GRAYSCALE)
    g = (g > 127).astype(np.uint8)[:1170,:1654]
    m = (m > 127).astype(np.uint8)
    tp = int((m&g).sum()); fp = int((m&~g.astype(bool)).sum()); fn = int((~m.astype(bool)&g).sum())
    prec = tp/max(1,tp+fp); rec_ = tp/max(1,tp+fn); iou = tp/max(1,tp+fp+fn)
    P.append(prec); R.append(rec_); IOU.append(iou)
    print(f"{iid:28s} {m.mean()*100:5.1f}% {g.mean()*100:5.1f}% {prec:6.3f} {rec_:7.3f} {iou:6.3f} {fp:8d} {fn:8d}")
print("-"*82)
print(f"mean precision {np.mean(P):.3f}  mean recall {np.mean(R):.3f}  mean IoU {np.mean(IOU):.3f}")
print(f"pages with precision < 0.5: {sum(1 for p in P if p<0.5)}/{len(P)}")
