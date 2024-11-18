import argparse
from typing import Optional

import cv2
import imageio
import numpy as np
from loguru import logger
from tqdm.auto import tqdm

from easy_dwpose import DWposeDetector


def load_video(input_path: str, max_video_len: Optional[int] = None) -> list[np.ndarray]:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Can't open video file {input_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

        if max_video_len and len(frames) >= max_video_len:
            break

    return frames, fps


def save_video(frames: list[np.ndarray], output_path: str, fps: int) -> None:    
    writer = imageio.get_writer(
        output_path, 
        fps=fps, 
        codec="libx264",
        output_params=['-pix_fmt', 'yuv420p']
    )

    for frame in frames:
        writer.append_data(frame)

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="assets/dance.mp4")
    parser.add_argument("--output_path", type=str, default="assets/skeleton.mp4")
    parser.add_argument("--max_video_len", type=int, default=None)
    args = parser.parse_args()

    detector = "checkpoints/yolox_l.onnx"
    pose_model = "checkpoints/dw-ll_ucoco_384.onnx"
    detector = DWposeDetector(detector, pose_model)
    logger.info(f"Loaded detector")

    logger.info(f"Starting inference on video {args.input}")
    video, fps = load_video(args.input, args.max_video_len)
    video = video[:10]

    result = []
    for frame in tqdm(video):
        result.append(detector(frame, output_type="np", include_hands=True, include_face=True))

    logger.info(f"Saving output video to {args.output_path}")
    save_video(result, args.output_path, fps)

    logger.info("Done")
