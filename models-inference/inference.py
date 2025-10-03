from fastapi import FastAPI
from ultralytics import YOLO
import os, cv2, glob

app = FastAPI()
model_weights = YOLO("models/best.pt")

@app.post("/cropToLicensePlates")
async def crop_to_license_plates(raw_data_dir: str, result_data_dir: str):
    try:
        from ultralytics import YOLO
    except Exception as e:
        print("Ultralytics not installed or failed to import. Please install ultralytics and try again.")
        return

    model = YOLO(model_weights)
    frame_paths = sorted(glob.glob(os.path.join(raw_data_dir, "*.png")))
    print(f"Found {len(frame_paths)} frames in {raw_data_dir}")
    crop_index = 0
    for i, fp in enumerate(frame_paths):
        img = cv2.imread(fp)
        if img is None: continue
        res = model.predict(source=img, conf=0.25, imgsz=1280, verbose=False)

        for r in res:
            print(len(r.boxes))
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                crop = img[y1:y2, x1:x2].copy()
                outp = os.path.join(result_data_dir, f"crop_{crop_index:06d}.png")
                cv2.imwrite(outp, crop)
                crop_index += 1
    print("Detection+crop finished. Crops saved to", result_data_dir)