# easy_dwpose

Recently, I tried to inference the DWPose (improved OpenPose) preprocessor for [Diffusers](https://github.com/huggingface/diffusers) and was shocked by how complicated it actually is!
So, I decided to change that!

The goal of Easy DWPose is to provide a generic, reliable, and easy-to-use interface for making skeletons for ControlNet.

Me: <a href="https://x.com/igorfeelippov"><img alt="X account" src="https://img.shields.io/twitter/url/https/twitter.com/diffuserslib.svg?style=social&label=Follow%20%40igorfeelippov"></a>

## Why you should use it :yum:

1. Easy installation!
2. Automatic checkpoint downloading.
3. Generic class to either import in Jupyter or to run on a video/images.
4. Code that is easy to read and modify.
5. Choose GPU for multi-gpu inference!
6. Custom drawing functions: convenient interface for modifying *how* you draw skeletons.

## Installation

### PIP

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
import torch
from PIL import Image

from easy_dwpose import DWposeDetector

# You can use a different GPU, e.g. "cuda:1"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
detector = DWposeDetector(device=device)
input_image = Image.open("assets/pose.png").convert("RGB")

skeleton = detector(input_image, output_type="pil", include_hands=True, include_face=True)
skeleton.save("skeleton.png")
```

<table align="center">
    <tr>
      <th align="center">Input</th>
      <th align="center">Output</th>
    </tr>
    <tr>
        <td align="center">
          <br />
          <img src="./assets/pose.png"/>
        </td>
        <td align="center">
          <br/>
          <img src="./assets/skeleton.png"/>
        </td>
    </tr>
</table>

### On a video

```bash
python scripts/inference_on_video.py --input assets/dance.mp4 --output_path result.mp4
```

<table align="center">
    <tr>
      <th align="center">Input</th>
      <th align="center">Output</th>
    </tr>
    <tr>
        <td align="center">
          <br />
          <img src="./assets/dance.gif"/>
        </td>
        <td align="center">
          <br/>
          <img src="./assets/skeleton.gif"/>
        </td>
    </tr>
</table>

### On a folder of images

```bash
python scripts/inference_on_folder.py --input assets/ --output_path results/
```

### Custom skeleton drawing

By default, we use standart skeleton drawing function but several projects change it (e.g. [MusePose](https://github.com/TMElyralab/MusePose)). Modify it or write your own from scratch!

```python
from PIL import Image

from easy_dwpose import DWposeDetector
from easy_dwpose.draw.musepose import draw_pose as draw_pose_musepose

detector = DWposeDetector(device="cpu")
input_image = Image.open("assets/pose.png").convert("RGB")

skeleton = detector(input_image, output_type="pil", draw_pose=draw_pose_musepose, draw_face=False)
skeleton.save("skeleton.png")
```

## Acknowledgement

We thank the original authors of the [DWPose](https://github.com/IDEA-Research/DWPose) for their incredible models!

Thanks for open-sourcing!
