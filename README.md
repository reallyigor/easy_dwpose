# easy_dwpose

Recently, I tried to inference the DWPose (improved OpenPose) preprocessor for [Diffusers](https://github.com/huggingface/diffusers) and was shocked by how complicated it actually is!
So, I decided to change that!

The goal of this project is to provide a generic, reliable, and easy-to-use interface for running DWPose.

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
from easy_dwpose import DWposeDetector

# You can use a different GPU, e.g. "cuda:1"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
detector = DWposeDetector(device=device)
input = Image.open("assets/pose.png").convert("RGB")

skeleton = detector(input, output_type="pil", include_hands=True, include_face=True)
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
python script/inference_on_video.py --input assets/dance.mp4 --output_path result.mp4
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
python script/inference_on_folder.py --input assets/ --output_path results/
```

## Acknowledgement

We thank the original authors of the [DWPose](https://github.com/IDEA-Research/DWPose) for their incredible models!

Thanks for open-sourcing!
