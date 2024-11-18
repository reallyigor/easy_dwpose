import argparse
from pathlib import Path

import cv2
from loguru import logger
from tqdm.auto import tqdm

from easy_dwpose import DWposeDetector

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default="assets/")
    parser.add_argument("--output_path", type=Path, default="results/")
    args = parser.parse_args()

    detector = "models/yolox_l.onnx"
    pose_model = "models/dw-ll_ucoco_384.onnx"
    detector = DWposeDetector(detector, pose_model)
    logger.info(f"Loaded detector")

    logger.info(f"Starting inference on folder {args.input}")
    files = sorted(args.input.iterdir())

    images = [file for file in files if file.suffix in [".jpg", ".jpeg", ".png"]]
    logger.info(f"Found {len(images)} images")

    args.output_path.mkdir(parents=True, exist_ok=True)

    for image_path in tqdm(images):
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        result = detector(image, output_type="np", include_hands=True, include_face=True)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

        output_path = str(image_path).replace(str(args.input), str(args.output_path))
        cv2.imwrite(output_path, result)

    logger.info("Done")
