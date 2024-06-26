from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from mmseg.apis import FIBDualInferencer, MMSegInferencer

from utils import stylized

CKP_DICT = {
    "mono": "./checkpoints/swin_40ke.pth",
    "dual": "./checkpoints/swin_40ke_dual.pth",
    "fuse": "./checkpoints/swin_40ke_fuse.pth",
}

parser = ArgumentParser()
parser.add_argument("--mode", choices=["mono", "dual", "fuse"], default="mono")
parser.add_argument("--checkpoint")
parser.add_argument("--img")
parser.add_argument("--img2")
parser.add_argument("--out-dir", default="./results")

args = parser.parse_args()

tmpdir = TemporaryDirectory()

if args.mode == "fuse":
    res = stylized(content=args.img2, style=args.img)
    im = cv2.imread(args.img, cv2.IMREAD_GRAYSCALE)
    args.img = str(Path(tmpdir.name) / Path(args.img).name)
    cv2.imwrite(args.img, np.array([res, im, im], dtype=np.uint8).transpose(1, 2, 0))

if args.checkpoint is None:
    args.checkpoint = CKP_DICT[args.mode]

if args.mode != "dual":
    inferencer = MMSegInferencer(
        model="./mmsegmentation/configs/swin/"
        "swin-tiny-patch4-window7-in1k-pre_upernet_4xb2-40ke_fib-512x512.py",
        weights=args.checkpoint,
    )

else:
    inferencer = FIBDualInferencer(
        model="./mmsegmentation/configs/swin/"
        "swin-tiny-patch4-window7_upernet_4xb2-40ke_fib-dual-512x512.py",
        weights=args.checkpoint,
    )

inferencer(
    (
        args.img
        if args.mode != "dual"
        else {
            "img_path": args.img,
            "img_path2": args.img2,
        }
    ),
    show=False,
    opacity=0.3,
    with_labels=False,
    out_dir=args.out_dir,
)

tmpdir.cleanup()
