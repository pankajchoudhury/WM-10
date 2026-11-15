# Robust wildlife detection and alerts: test code, data and weights

With this repository, you can re-run the paper's real-world tests on:

- **real images:** detection accuracy, plus the ALERT/LOG decision for each image;
- **a real camera-trap video:** the alert timeline.

Four trained detectors are provided: YOLO26n and YOLO26m, each with **normal** fine-tuning (clean images only) and **robust** fine-tuning (each training image plus one synthetic night, IR, fog, motion-blur or occlusion copy).

No training code or training data is needed to run the tests.

## Examples

**Representative examples of the ten animal classes in WM-10-Original.** Cow and Goat are grouped under the Domestic class.

![Ten classes](assets/classes.jpg)

**Original image and five environmental transformations used in WM-10-Robust.** The bounding-box annotations remain unchanged.

![Synthetic conditions](assets/conditions.jpg)

**Condition-wise species-level detection performance (mAP50:95) of (a) YOLO26n and (b) YOLO26m.** Grey bars: normal fine-tuning (Normal FT). Blue bars: robust fine-tuning (Robust FT).

![Synthetic test results](assets/synthetic_results.png)

**Qualitative images from the WM-10-CameraTrap test set, predicted by normal and robust fine-tuned YOLO26m.** Green boxes are correct species predictions. Red dashed boxes are incorrect predictions.

![Real-world examples](assets/qualitative_real.jpg)

**Video-based wildlife alert evaluation using YOLO26m.**
- The upper panels compare normal and robust fine-tuning at t = 34.0 s.
- The lower panel shows the ALERT and LOG decisions over time, after two-out-of-three-frame smoothing.
- The dark bar shows when the animal is present, and the dotted line marks the selected frame.

![Video alert example](assets/video_alert.png)

## Get the data, weights and video

The images, video and weights (about 196 MB) are in this Google Drive folder:
**[wildlife-alert-data](https://drive.google.com/drive/folders/110E8umWRAlWJeEi400wlHsWi5FM00Wo7?usp=sharing)**

1. Open the link and click **Download** (top right). Google Drive sends the folder as a zip.
2. Unzip it.
3. Place the three folders `data/`, `videos/` and `weights/` next to the test scripts.
4. Run `python test_real_images.py --check` to confirm that the data is in the right place.

The layout should be:

```
data/WM-10-CameraTrap/
  images/{day,fog,night,ir}/   183 real images (day 33, fog 54, night 53, IR 43)
  labels/{day,fog,night,ir}/   YOLO labels (class x_center y_center width height, normalised)
  conditions.csv               image, class, condition, number of boxes
videos/
  GF5LRYXabHU.mp4              50 s night/IR camera-trap clip (leopard, then tiger)
  GF5LRYXabHU_groundtruth.csv  when a dangerous animal is on screen (start_s, end_s)
  sources.csv                  source of the video
weights/
  yolo26n_normal.pt  yolo26n_robust.pt  yolo26m_normal.pt  yolo26m_robust.pt
```

The repository itself holds:

```
test_real_images.py   real-image test: mAP per condition + ALERT/LOG confusion matrix
test_video.py         video test: ALERT/LOG per frame, smoothed alert timeline, scores
expected_results/     the paper's numbers; the scripts print them next to yours
assets/               the example figures above
```

## Quick start (Google Colab)

First download the Drive folder as a zip (see above). In Colab, upload the zip with the Files panel on the left, then run:

```python
!git clone <this repository URL> repo
%cd repo
!pip install -q -r requirements.txt
!unzip -q -o "/content/*.zip" -d /content/dl
!D=$(dirname "$(find /content/dl -type d -name data | head -1)"); cp -r "$D/data" "$D/videos" "$D/weights" .
!python test_real_images.py --check          # data check only (no model is run)
!python test_real_images.py --save-images    # 4 models on the 183 real images
!python test_video.py --save-video           # 4 models on the video
```

On your own machine, run the same commands without `!` (Python 3.10 or newer). A GPU is optional.

Options:
- `--device cpu` or `--device 0` chooses the device. By default, the GPU is used if there is one.
- `--weights weights/yolo26m_robust.pt` tests one model only.
- `python <script> -h` lists all options.

Outputs go to `results/real_images/` and `results/video/`:
- `--save-images` writes every test image with its predicted boxes. A banner shows the true and predicted decision: green if they match, red if not.
- `--save-video` writes one annotated clip per model.

## Classes and decisions

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

- An image or video frame is **ALERT** if any ALERT class is detected with confidence at least 0.25. Otherwise it is **LOG**.
- If several animals are detected, ALERT wins over LOG.
- In the video, the alert is ON when at least 2 of the last 3 sampled frames (5 frames per second) are ALERT.

## Expected results

These are the results reported in the paper. The scripts print them in brackets next to your values.

### Real-world detection performance (mAP50:95) on WM-10-CameraTrap (`test_real_images.py`)

The value is the mean over the day, fog, night and IR images. N and R denote normal and robust fine-tuning.

| Detector | N | R (Δ) |
|---|---|---|
| YOLO26n | 0.173 | 0.203 (+0.030) |
| YOLO26m | 0.200 | 0.219 (+0.019) |

### Wildlife alert decisions on 183 real-world images (`test_real_images.py`)

Columns show the predicted actions, grouped by the true action.

| Detector | FT | True ALERT (112): ALERT | True ALERT (112): LOG | True LOG (71): ALERT | True LOG (71): LOG |
|---|---|---|---|---|---|
| YOLO26n | N | 44 | 68 | 13 | 58 |
| | R | 67 | 45 | 2 | 69 |
| YOLO26m | N | 50 | 62 | 3 | 68 |
| | R | 70 | 42 | 1 | 70 |

### Video-based alert evaluation, YOLO26m (`test_video.py`)

The 49.9 s camera-trap video is evaluated with 250 frames sampled at 5 fps. It shows a leopard, then a tiger. The alert is on when an ALERT prediction appears in at least 2 of the last 3 frames.

| Detector | FT | Correct alert time (s) | Correct alert coverage |
|---|---|---|---|
| YOLO26m | N | 22.6 | 45.7% |
| | R | 39.4 | 79.8% |

The paper's image results were computed on an NVIDIA P100 GPU and its video results on Google Colab, both with `ultralytics==8.4.175`. A different GPU, a CPU, or another library version can change the values in the third decimal place. It can also change single image decisions.

## Data notes

- The WM-10-CameraTrap images are frames from publicly available wildlife videos (camera-trap, CCTV and other footage). None of them were used for training.
- The boxes were drawn by hand.
- The test video comes from a public source, listed in `videos/sources.csv`.
- These files are shared only so that the paper's results can be checked.
