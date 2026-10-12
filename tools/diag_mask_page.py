import glob, json, sys
from pathlib import Path
import cv2, numpy as np
sys.stdout.reconfigure(encoding="utf-8")
IDS = ["ARMS/000","AkkeraKanjinchou/000","AisazuNihaIrarenai/005","AisazuNihaIrarenai/004"]
out = Path("deliverables/diag/maskpages"); out.mkdir(parents=True, exist_ok=True)
for path in sorted(glob.glob("outputs/translation/spec004-vilao/*/*/*/metadata.json")):
    rec = json.loads(Path(path).read_text(encoding="utf-8"))
    if rec["image_id"] not in IDS: continue
    page = Path(path).parent
    orig = cv2.imread(str(page/"original.png")); m = cv2.imread(str(page/"mask.png"), cv2.IMREAD_GRAYSCALE)
    ov = orig.copy(); ov[m>0] = (0,0,255)
    s = 0.42
    row = np.hstack([cv2.resize(x,None,fx=s,fy=s) for x in (orig, ov, cv2.cvtColor(m,cv2.COLOR_GRAY2BGR))])
    cv2.imwrite(str(out/f"{rec['image_id'].replace('/','_')}.png"), row)
    print(rec["image_id"], "mask%", round(m.mean()/255*100,1))
