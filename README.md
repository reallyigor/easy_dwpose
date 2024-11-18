# easy_dwpose

The goal of this project is to provide a generic, reliable and easy-to-use interface to run DWPose preprocessor for the OpenPose ControlNet.

## Why you should use it :yum:

1. Easy installation!
2. Automatic checkpoint downloading.
3. Generic class to either import in Jupyter or to run on a video/folder of images.
4. Batch inference.
5. Code that is easy to read and modify.
6. Choose GPU for multi-gpu inference!
7. Custom drawing functions: easy modify *how* you draw.

## Installation

### Pip

```bash
pip install easy-dwpose
```

### From source

```bash
git clone git@github.com:reallyigor/easy_dwpose.git
cd easy_dwpose
pip install -e .
```

## Quickstart

### In you own .py scrip or in Jupyter

```python
from easy_dwpose import DWposeDetector

# You can use a different GPU, e.g. "cuda:1"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
detector = DWposeDetector(device=device)
input = Image.open("assets/pose.png").convert("RGB")

skeleton = detector(input, output_type="pil", include_hands=True, include_face=True)
skeleton.save("skeleton.png")
```

![skeleton](./assets/skeleton.png)

### On a video

```bash
python script/inference_on_video.py --input assets/dance.mp4 --output_path result.mp4
```

result.mp4

![result](https://github.com/user-attachments/assets/2be469d8-b177-47a6-9195-34cb7ce764dc)

### On a folder of images

```bash
python script/inference_on_folder.py --input assets/ --output_path results/
```

## Acknowledgement

We thank the original authors of the [DWPose](https://github.com/IDEA-Research/DWPose) for their incredible models!

Thanks for open-sourcing!
