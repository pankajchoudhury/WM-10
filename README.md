# Robust wildlife detection and alerts: test code, data and weights

This repository lets you re-run the real-world tests of the paper. It holds:

- two test scripts (real images and video);
- four trained detectors (YOLO26n and YOLO26m, each with normal and robust fine-tuning);
- the real-world image set WM-10-CameraTrap (183 annotated images);
- one camera-trap test video with its ground truth.

No training code or training data is needed to run the tests.

## Contents

```
test_real_images.py          real-image test: mAP per condition + ALERT/LOG confusion matrix
test_video.py                video test: per-frame ALERT/LOG, smoothed alert timeline, scores
weights/
  yolo26n_normal.pt          YOLO26n, normal fine-tuning (clean images only)
  yolo26n_robust.pt          YOLO26n, robust fine-tuning (clean + one synthetic-condition copy per image)
  yolo26m_normal.pt          YOLO26m, normal fine-tuning
  yolo26m_robust.pt          YOLO26m, robust fine-tuning
data/WM-10-CameraTrap/
  images/{day,fog,night,ir}/ 183 real images (day 33, fog 54, night 53, IR 43)
  labels/{day,fog,night,ir}/ YOLO labels (class x_center y_center width height, normalised)
  conditions.csv             image, class, condition, number of boxes
videos/
  GF5LRYXabHU.mp4            50 s night/IR camera-trap clip (leopard, then tiger)
  GF5LRYXabHU_groundtruth.csv  when a dangerous animal is on screen (start_s, end_s)
  sources.csv                source of the video
expected_results/            our numbers, printed next to yours by the scripts
```

## Classes and decisions

The detectors have 10 classes:

| ID | Class | Decision |
|---|---|---|
| 0 | Domestic (cow, goat) | LOG |
| 1 | Elephant | ALERT |
| 2 | Pygmy Hog | LOG |
| 3 | Buffalo | ALERT |
| 4 | Monkey | LOG |
| 5 | Deer | LOG |
| 6 | Leopard | ALERT |
| 7 | Rhino | ALERT |
| 8 | Royal Bengal Tiger | ALERT |
| 9 | Asiatic Black Bear | ALERT |

- An image (or video frame) is **ALERT** if any ALERT class is detected with confidence at least 0.25. Otherwise it is **LOG**.
- If several animals are detected, ALERT wins over LOG.
- In the video, the alert is ON when at least 2 of the last 3 sampled frames (5 frames per second) are ALERT.

## Setup

Python 3.10 or newer. A GPU is optional. On a CPU, the image test takes a few minutes and the video test takes longer.

```bash
pip install -r requirements.txt
```

The paper used `ultralytics==8.4.175`. Other versions may change the numbers slightly.

## Run

```bash
python test_real_images.py --check    # check the data only (no model is run)
python test_real_images.py            # all four models on the 183 real images
python test_video.py                  # all four models on the video
python test_video.py --save-video     # also write one annotated video per model
```

Options:
- `--device cpu` or `--device 0` chooses the device. By default, the GPU is used if there is one.
- `--weights weights/yolo26m_robust.pt` tests one model only.
- `python <script> -h` lists all options.

Results go to `results/real_images/` and `results/video/`.

### Google Colab

```python
!git clone <this repository URL> repo
%cd repo
!pip install -q -r requirements.txt
!python test_real_images.py
!python test_video.py
```

## Expected results

### Real images: mAP50-95 per condition

| Model | Day | Fog | Night | IR | Mean |
|---|---|---|---|---|---|
| YOLO26n normal | 0.270 | 0.177 | 0.101 | 0.142 | 0.173 |
| YOLO26n robust | 0.283 | 0.230 | 0.142 | 0.155 | 0.203 |
| YOLO26m normal | 0.315 | 0.202 | 0.133 | 0.150 | 0.200 |
| YOLO26m robust | 0.287 | 0.256 | 0.163 | 0.171 | 0.219 |

These use the default Ultralytics validation settings, with image size 640 and batch size 32.

### Real images: ALERT / LOG confusion

There are 183 images: 112 need an ALERT and 71 are LOG.

| Model | True ALERT → ALERT | True ALERT → LOG (missed) | True LOG → ALERT (false) | True LOG → LOG |
|---|---|---|---|---|
| YOLO26n normal | 44 | 68 | 13 | 58 |
| YOLO26n robust | 67 | 45 | 2 | 69 |
| YOLO26m normal | 50 | 62 | 3 | 68 |
| YOLO26m robust | 70 | 42 | 1 | 70 |

### Video: GF5LRYXabHU

A dangerous animal is on screen for 49.4 s of the clip. The table shows the smoothed alert.

| Model | Alert ON (s) | Correct (s) | False (s) | Coverage | Alert onsets |
|---|---|---|---|---|---|
| YOLO26n normal | 29.2 | 28.6 | 0.6 | 57.9% | 6 |
| YOLO26n robust | 39.6 | 39.0 | 0.6 | 78.9% | 5 |
| YOLO26m normal | 22.6 | 22.6 | 0.0 | 45.7% | 4 |
| YOLO26m robust | 39.4 | 39.4 | 0.0 | 79.8% | 3 |

Our image results were computed on an NVIDIA P100 GPU with Ultralytics 8.4.175. A different GPU, a CPU, or another library version can change the values in the third decimal place. It can also change single image decisions.

## Data notes

- The WM-10-CameraTrap images are frames from publicly available wildlife videos (camera-trap, CCTV and other footage). None of them were used for training.
- The boxes were drawn by hand.
- The test video comes from a public source, listed in `videos/sources.csv`.
- These files are shared only so that the paper's results can be checked.
