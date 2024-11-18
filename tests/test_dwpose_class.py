from PIL import Image

from easy_dwpose import DWposeDetector
from easy_dwpose.draw.mimic_motion import draw_pose as draw_pose_mimic_motion
from easy_dwpose.draw.musepose import draw_pose as draw_pose_musepose

SAVE_SIZE = (256, 448)


def test_forward():
    detector_path = "checkpoints/yolox_l.onnx"
    pose_model_path = "checkpoints/dw-ll_ucoco_384.onnx"

    detector = DWposeDetector(detector_path, pose_model_path)
    input = Image.open("assets/pose.png").convert("RGB")
    result = detector(input, output_type="pil", include_hands=True, include_face=True)

    return result


def test_replace_drawing_musepose():
    detector_path = "checkpoints/yolox_l.onnx"
    pose_model_path = "checkpoints/dw-ll_ucoco_384.onnx"

    detector = DWposeDetector(detector_path, pose_model_path)
    input = Image.open("assets/pose.png").convert("RGB")
    result = detector(input, output_type="pil", draw_pose=draw_pose_musepose, draw_face=False)

    return result


def test_replace_drawing_mimic_motion():
    detector_path = "checkpoints/yolox_l.onnx"
    pose_model_path = "checkpoints/dw-ll_ucoco_384.onnx"

    detector = DWposeDetector(detector_path, pose_model_path)
    input = Image.open("assets/pose.png").convert("RGB")
    result = detector(input, output_type="pil", draw_pose=draw_pose_mimic_motion)

    return result


if __name__ == "__main__":
    res_test_forward = test_forward()
    res_test_forward.resize(SAVE_SIZE).save("test_forward.png")

    res_test_replace_drawing_musepose = test_replace_drawing_musepose()
    res_test_replace_drawing_musepose.resize(SAVE_SIZE).save("test_replace_drawing_musepose.png")

    res_test_replace_drawing_mimic_motion = test_replace_drawing_mimic_motion()
    res_test_replace_drawing_mimic_motion.resize(SAVE_SIZE).save("test_replace_drawing_mimic_motion.png")
