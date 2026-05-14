"""Quick API test: posts sample images to the backend and saves comparison grids."""
import base64
from pathlib import Path

import cv2
import numpy as np
import requests

test_imgs = [
    "database/processed_lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg",
    "database/processed_lfw/Aaron_Patterson/Aaron_Patterson_0001.jpg",
    "database/processed_lfw/Aaron_Peirsol/Aaron_Peirsol_0002.jpg",
]

Path("outputs/ui_test").mkdir(parents=True, exist_ok=True)

for img_path in test_imgs:
    name = Path(img_path).stem
    with open(img_path, "rb") as f:
        r = requests.post("http://localhost:8000/analyze", files={"file": f})
    d = r.json()
    print(
        f"{name}: detected={d['detected_face']} occluded={d['is_occluded']} "
        f"region={d['occlusion_region']} ratio={d['occlusion_ratio']:.3f} "
        f"predicted={d['predicted_name']} conf={d['confidence']:.3f} fps={d['fps']:.1f}"
    )

    panels = []
    labels = ["Input (occluded)", "Reconstructed", "Original (gallery)"]
    for key in ["input_image_base64", "reconstructed_image_base64", "original_image_base64"]:
        b64 = d.get(key, "")
        if b64:
            img_bytes = base64.b64decode(b64)
            arr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            panels.append(img)
        else:
            panels.append(np.zeros((112, 112, 3), dtype=np.uint8))

    # Resize all panels to the same height
    h = max(p.shape[0] for p in panels)
    resized = []
    for i, p in enumerate(panels):
        if p.shape[0] != h:
            scale = h / p.shape[0]
            p = cv2.resize(p, (int(p.shape[1] * scale), h))
        # Add label bar
        bar = np.zeros((24, p.shape[1], 3), dtype=np.uint8)
        cv2.putText(bar, labels[i], (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        resized.append(np.vstack([bar, p]))

    grid = np.hstack(resized)
    out = f"outputs/ui_test/{name}_comparison.png"
    cv2.imwrite(out, grid)
    print(f"  -> Saved {out}")
