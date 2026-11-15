#!/usr/bin/env python3
# ================================================================
#   REAL-IMAGE TEST  -  WM-10-CameraTrap (183 real frames: day 33, fog 54, night 53, IR 43)
#
#   For every model in --weights:
#     1. Detection: species-level mAP50 and mAP50-95 per condition (Ultralytics val, default
#        val settings), and the mean over the four conditions.
#     2. Alert: one ALERT / LOG decision per image at confidence >= 0.25.
#        ALERT if any dangerous class is detected (Elephant, Leopard, Rhino, Royal Bengal Tiger,
#        Asiatic Black Bear, Buffalo), otherwise LOG. Compared with the decision implied by the labels.
#
#   Output (--out):
#     detection.csv                    mAP50 and mAP50-95 per model and condition (+ mean)
#     alert_confusion.csv              true ALERT / LOG vs predicted ALERT / LOG per model
#     <model>_image_decisions.csv      one row per image: true and predicted classes and decisions
#   If expected_results/real_images.csv exists, the paper's numbers are printed next to yours.
#
#   Examples:
#     python test_real_images.py                                   # the four models in weights/
#     python test_real_images.py --weights weights/yolo26m_robust.pt --device cpu
#     python test_real_images.py --check                           # check the data only, no model is run
# ================================================================

import argparse
import csv
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAMES = ['Domestic', 'Elephant', 'Pygmy Hog', 'Buffalo', 'Monkey', 'Deer', 'Leopard', 'Rhino',
         'Royal Bengal Tiger', 'Asiatic Black Bear']
DANGER = {'Elephant', 'Leopard', 'Rhino', 'Royal Bengal Tiger', 'Asiatic Black Bear', 'Buffalo'}
CONDITIONS = ['day', 'fog', 'night', 'ir']
EXPECTED_N = {'day': 33, 'fog': 54, 'night': 53, 'ir': 43}
MODELS = ['yolo26n_normal', 'yolo26n_robust', 'yolo26m_normal', 'yolo26m_robust']
IMG_EXT = {'.jpg', '.jpeg', '.png'}


def parse_args():
    p = argparse.ArgumentParser(description='Test the detectors on the real-world image set (WM-10-CameraTrap).',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--weights', nargs='+', type=Path, default=[ROOT / 'weights' / f'{m}.pt' for m in MODELS])
    p.add_argument('--data', type=Path, default=ROOT / 'data' / 'WM-10-CameraTrap')
    p.add_argument('--out', type=Path, default=ROOT / 'results' / 'real_images')
    p.add_argument('--conf', type=float, default=0.25, help='confidence threshold for the ALERT / LOG decision')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--batch', type=int, default=32, help='batch size (as in the paper)')
    p.add_argument('--device', default=None, help="'0' for the first GPU or 'cpu' (default: GPU if available)")
    p.add_argument('--check', action='store_true', help='only check the data folder; no model is run')
    return p.parse_args()


def read_labels(lbl):
    """Class names in one YOLO label file."""
    out = set()
    for ln in lbl.read_text().splitlines():
        if ln.strip():
            c = int(float(ln.split()[0]))
            if not 0 <= c < len(NAMES):
                raise SystemExit(f'{lbl}: class id {c} is not one of the {len(NAMES)} classes')
            out.add(NAMES[c])
    return out


def check_data(data):
    """Every image has a label file; counts match the paper. Returns {condition: [image paths]}, {stem: classes}."""
    images, truth = {}, {}
    for cond in CONDITIONS:
        imgs = sorted(p for p in (data / 'images' / cond).glob('*') if p.suffix.lower() in IMG_EXT)
        for p in imgs:
            lbl = data / 'labels' / cond / (p.stem + '.txt')
            if not lbl.exists():
                raise SystemExit(f'No label file for {p}')
            truth[p.stem] = read_labels(lbl)
        if len(imgs) != EXPECTED_N[cond]:
            print(f'  WARNING: {cond} has {len(imgs)} images, the paper used {EXPECTED_N[cond]}')
        images[cond] = imgs
    n_alert = sum(bool(t & DANGER) for t in truth.values())
    print(f'Data: {sum(map(len, images.values()))} images '
          f'({", ".join(f"{c} {len(v)}" for c, v in images.items())}); '
          f'{n_alert} need an ALERT, {len(truth) - n_alert} are LOG')
    return images, truth


def write_yaml(path, data, cond):
    path.write_text('\n'.join(
        [f'path: {data.resolve().as_posix()}', f'train: images/{cond}', f'val: images/{cond}',
         f'test: images/{cond}', '', f'nc: {len(NAMES)}', 'names:']
        + [f'  {i}: {n}' for i, n in enumerate(NAMES)]) + '\n')


def detection(model, tag, data, tmp, args):
    """mAP50 and mAP50-95 per condition."""
    res = {}
    for cond in CONDITIONS:
        yaml = tmp / f'data_{cond}.yaml'
        write_yaml(yaml, data, cond)
        r = model.val(data=str(yaml), split='test', imgsz=args.imgsz, batch=args.batch, device=args.device,
                      workers=0, plots=False, verbose=False, project=str(tmp), name=f'{tag}_{cond}', exist_ok=True)
        res[cond] = (float(r.box.map50), float(r.box.map))
        print(f'  {cond:<6} mAP50 {r.box.map50:.3f}  mAP50-95 {r.box.map:.3f}')
    return res


def alerts(model, images, truth, args):
    """ALERT / LOG per image; images in name order, in batches of --batch (as in the paper)."""
    imgs = sorted((p for c in CONDITIONS for p in images[c]), key=lambda p: p.name)
    cond_of = {p.stem: c for c in CONDITIONS for p in images[c]}
    cm, rows = {'alert_alert': 0, 'alert_log': 0, 'log_alert': 0, 'log_log': 0}, []
    for start in range(0, len(imgs), args.batch):
        for res in model.predict([str(p) for p in imgs[start:start + args.batch]], conf=args.conf,
                                 imgsz=args.imgsz, device=args.device, verbose=False):
            stem = Path(res.path).stem
            det = sorted(((model.names[int(c)], float(s)) for c, s in zip(res.boxes.cls, res.boxes.conf)),
                         key=lambda x: -x[1])
            t = 'alert' if truth[stem] & DANGER else 'log'
            p = 'alert' if {c for c, _ in det} & DANGER else 'log'
            cm[f'{t}_{p}'] += 1
            rows.append([stem, cond_of[stem], '; '.join(sorted(truth[stem])),
                         '; '.join(f'{c} {s:.2f}' for c, s in det), t.upper(), p.upper()])
    print(f"  alert  true ALERT -> ALERT {cm['alert_alert']}, LOG {cm['alert_log']} | "
          f"true LOG -> ALERT {cm['log_alert']}, LOG {cm['log_log']}")
    return cm, rows


def write_csv(path, header, rows):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    args = parse_args()
    images, truth = check_data(args.data)
    if args.check:
        return
    import torch
    from ultralytics import YOLO
    args.device = args.device or ('0' if torch.cuda.is_available() else 'cpu')
    args.out.mkdir(parents=True, exist_ok=True)
    tmp = args.out / '_ultralytics'

    det_rows, cm_rows, summary = [], [], {}
    for w in args.weights:
        tag = w.stem
        model = YOLO(str(w))
        if [model.names[i] for i in range(len(model.names))] != NAMES:
            raise SystemExit(f'{w}: unexpected classes {model.names}')
        print(f'\n#### {tag}')
        det = detection(model, tag, args.data, tmp, args)
        mean = sum(v[1] for v in det.values()) / len(det)
        cm, rows = alerts(model, images, truth, args)
        det_rows += [[tag, c, round(v[0], 4), round(v[1], 4)] for c, v in det.items()]
        det_rows.append([tag, 'mean', round(sum(v[0] for v in det.values()) / len(det), 4), round(mean, 4)])
        cm_rows.append([tag, *cm.values()])
        write_csv(args.out / f'{tag}_image_decisions.csv',
                  ['image', 'condition', 'true_classes', 'pred_classes (conf)', 'true_decision', 'pred_decision'], rows)
        summary[tag] = ({c: v[1] for c, v in det.items()}, mean, cm)
    shutil.rmtree(tmp, ignore_errors=True)
    write_csv(args.out / 'detection.csv', ['model', 'condition', 'mAP50', 'mAP50-95'], det_rows)
    write_csv(args.out / 'alert_confusion.csv',
              ['model', 'true_ALERT_pred_ALERT', 'true_ALERT_pred_LOG', 'true_LOG_pred_ALERT', 'true_LOG_pred_LOG'],
              cm_rows)

    exp_file = ROOT / 'expected_results' / 'real_images.csv'
    exp = {r['model']: r for r in csv.DictReader(open(exp_file))} if exp_file.exists() else {}
    print('\n' + '=' * 96)
    print('Detection, mAP50-95 (paper value in brackets)')
    print(f"  {'model':<16}" + ''.join(f'{c:>16}' for c in CONDITIONS + ['mean']))
    for tag, (per, mean, _) in summary.items():
        e = exp.get(tag, {})
        cells = [f"{v:.3f} ({float(e[c]):.3f})" if c in e else f'{v:.3f}' for c, v in [*per.items(), ('mean', mean)]]
        print(f'  {tag:<16}' + ''.join(f'{x:>16}' for x in cells))
    print('\nAlert decisions on 183 images (paper value in brackets)')
    print(f"  {'model':<16}{'ALERT->ALERT':>16}{'ALERT->LOG':>16}{'LOG->ALERT':>16}{'LOG->LOG':>16}")
    for tag, (_, _, cm) in summary.items():
        e = exp.get(tag, {})
        print(f'  {tag:<16}' + ''.join(f"{f'{v} ({e[k]})' if k in e else str(v):>16}" for k, v in cm.items()))
    print(f'\nSaved to {args.out.resolve()}')


if __name__ == '__main__':
    main()
