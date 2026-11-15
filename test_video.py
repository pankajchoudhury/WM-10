#!/usr/bin/env python3
# ================================================================
#   VIDEO ALERT TEST  -  run the detectors on a real camera-trap video and score the ALERT timeline
#
#   The video is sampled at --fps (5 frames per second); every model sees exactly the same frames.
#   Frame decision : ALERT if any dangerous class is detected at confidence >= 0.25, otherwise LOG.
#                    Dangerous: Elephant, Leopard, Rhino, Royal Bengal Tiger, Asiatic Black Bear, Buffalo.
#   Smoothed alert : ON when the frame decision is ALERT in at least 2 of the last 3 sampled frames.
#   With a ground-truth file (rows: video, animal, start_s, end_s, note) the smoothed alert is scored:
#     alert_s    seconds with the alert ON
#     correct_s  seconds with the alert ON while a dangerous animal is on screen
#     false_s    seconds with the alert ON while no dangerous animal is on screen
#     present_s  seconds with a dangerous animal on screen
#     coverage   correct_s / present_s
#     onsets     how many times the alert switches on (fewer = less flicker)
#
#   Output (--out):
#     <video>_detections.csv     one row per box: frame, time, model, class, confidence, x1 y1 x2 y2
#     <video>_frames.csv         one row per sampled frame: ground truth + frame / smoothed decision per model
#     <video>_summary.csv        the scores above, one row per model
#     <video>_<model>.mp4        (--save-video) the sampled frames with boxes and the smoothed decision
#   If expected_results/video_<video>.csv exists, our reference numbers are printed next to yours.
#
#   Examples:
#     python test_video.py                                         # videos/GF5LRYXabHU.mp4, four models
#     python test_video.py --save-video --device cpu
#     python test_video.py --video my_clip.mp4 --gt none           # any video, no scoring
# ================================================================

import argparse
import csv
from collections import deque
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
DANGER = {'Elephant', 'Leopard', 'Rhino', 'Royal Bengal Tiger', 'Asiatic Black Bear', 'Buffalo'}
MODELS = ['yolo26n_normal', 'yolo26n_robust', 'yolo26m_normal', 'yolo26m_robust']
ORANGE, GREY = (1, 97, 230), (122, 122, 122)          # BGR


def parse_args():
    p = argparse.ArgumentParser(description='Run the detectors on a video and score the ALERT timeline.',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--video', type=Path, default=ROOT / 'videos' / 'GF5LRYXabHU.mp4')
    p.add_argument('--gt', default=None, help="ground-truth CSV (default: <video>_groundtruth.csv next to the "
                                              "video); 'none' = no scoring")
    p.add_argument('--weights', nargs='+', type=Path, default=[ROOT / 'weights' / f'{m}.pt' for m in MODELS])
    p.add_argument('--out', type=Path, default=ROOT / 'results' / 'video')
    p.add_argument('--fps', type=float, default=5, help='sampled frames per second')
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--k', type=int, default=2, help='smoothing: ALERT in at least K ...')
    p.add_argument('--n', type=int, default=3, help='... of the last N sampled frames')
    p.add_argument('--device', default=None, help="'0' for the first GPU or 'cpu' (default: GPU if available)")
    p.add_argument('--save-video', action='store_true', help='write one annotated video per model')
    return p.parse_args()


def load_gt(args):
    if args.gt == 'none':
        return None
    path = Path(args.gt) if args.gt else args.video.with_name(f'{args.video.stem}_groundtruth.csv')
    if not path.exists():
        print(f'No ground truth ({path}): detections are saved but not scored')
        return None
    return [(float(r['start_s']), float(r['end_s'])) for r in csv.DictReader(open(path)) if r['animal'] in DANGER]


def draw(frame, dets, on, tag):
    img = frame.copy()
    for c, s, b in dets:
        col = ORANGE if c in DANGER else GREY
        x1, y1, x2, y2 = map(int, b)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 3)
        cv2.putText(img, f'{c} {s:.2f}', (x1 + 4, max(y1 - 8, 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
    cv2.putText(img, f"{tag}: {'ALERT' if on else 'LOG'}", (12, img.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                ORANGE if on else GREY, 3)
    return img


def score(times, smooth, gt, dt):
    inside = [any(s <= t < e for s, e in gt) for t in times]
    correct = sum(a and i for a, i in zip(smooth, inside))
    return {'alert_s': round(sum(smooth) * dt, 1), 'correct_s': round(correct * dt, 1),
            'false_s': round(sum(a and not i for a, i in zip(smooth, inside)) * dt, 1),
            'present_s': round(sum(inside) * dt, 1), 'coverage': round(correct / max(sum(inside), 1), 3),
            'onsets': sum(1 for i, a in enumerate(smooth) if a and (i == 0 or not smooth[i - 1]))}


def main():
    args = parse_args()
    gt = load_gt(args)
    import torch
    from ultralytics import YOLO
    args.device = args.device or ('0' if torch.cuda.is_available() else 'cpu')
    args.out.mkdir(parents=True, exist_ok=True)
    vid = args.video.stem
    models = {w.stem: YOLO(str(w)) for w in args.weights}

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f'Cannot open {args.video}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, round(fps / args.fps))
    dt = step / fps                                            # seconds per sampled frame
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writers = {t: cv2.VideoWriter(str(args.out / f'{vid}_{t}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps / step, (W, H))
               for t in models} if args.save_video else {}

    times, idxs, dets = [], [], []
    raw = {t: [] for t in models}
    smooth = {t: [] for t in models}
    window = {t: deque(maxlen=args.n) for t in models}
    idx = 0
    while cap.grab():
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            t = round(idx / fps, 3)
            times.append(t); idxs.append(idx)
            for tag, m in models.items():
                r = m.predict(frame, conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)[0]
                fd = [(m.names[int(c)], float(s), b) for c, s, b in
                      zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist())]
                dets += [[vid, idx, t, tag, c, round(s, 4)] + [round(x, 1) for x in b] for c, s, b in fd]
                a = any(c in DANGER for c, _, _ in fd)
                window[tag].append(a)
                raw[tag].append(a)
                smooth[tag].append(sum(window[tag]) >= args.k)
                if tag in writers:
                    writers[tag].write(draw(frame, fd, smooth[tag][-1], tag))
        idx += 1
    cap.release()
    for w in writers.values():
        w.release()
    print(f'{vid}: {len(times)} sampled frames of {idx} (video {fps:.2f} fps, every {step}th frame), '
          f'{len(dets)} boxes')

    with open(args.out / f'{vid}_detections.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['video', 'frame_idx', 'time_s', 'model', 'class', 'conf', 'x1', 'y1', 'x2', 'y2'])
        w.writerows(dets)
    with open(args.out / f'{vid}_frames.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['frame_idx', 'time_s', 'danger_on_screen']
                   + [f'{t}_{k}' for t in models for k in ('frame', 'smoothed')])
        for i, t in enumerate(times):
            g = '' if gt is None else int(any(s <= t < e for s, e in gt))
            w.writerow([idxs[i], t, g] + [x for tag in models for x in
                                          ('ALERT' if raw[tag][i] else 'LOG', 'ALERT' if smooth[tag][i] else 'LOG')])
    if gt is None:
        print(f'Saved to {args.out.resolve()}')
        return

    res = {tag: score(times, smooth[tag], gt, dt) for tag in models}
    keys = list(next(iter(res.values())))
    with open(args.out / f'{vid}_summary.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['video', 'model'] + keys)
        w.writerows([vid, tag] + list(s.values()) for tag, s in res.items())

    exp_file = ROOT / 'expected_results' / f'video_{vid}.csv'
    exp = {r['model']: r for r in csv.DictReader(open(exp_file))} if exp_file.exists() else {}
    print(f'\nSmoothed ALERT ({args.k} of last {args.n} frames), reference value in brackets')
    print(f"  {'model':<16}" + ''.join(f'{k:>16}' for k in keys))
    for tag, s in res.items():
        e = exp.get(tag, {})
        print(f'  {tag:<16}' + ''.join(f"{f'{v} ({e[k]})' if k in e else str(v):>16}" for k, v in s.items()))
    print(f'\nSaved to {args.out.resolve()}')


if __name__ == '__main__':
    main()
