import os
import shutil
from glob import glob
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from utils import stylized

IN_DIRS = ("./data/se_img/", "./data/bse_img/")
OUT_DIR = "./data/fuse/"


def main():
    if Path(OUT_DIR).is_dir():
        shutil.rmtree(OUT_DIR)
    os.mkdir(OUT_DIR)

    for ds in ("train", "val"):
        os.mkdir(f"{OUT_DIR}{ds}")
        ims1, ims2 = (glob(f"{in_dir}{ds}/*") for in_dir in IN_DIRS)
        for im1, im2 in tqdm(list(zip(ims1, ims2)), desc=f"Generating {ds} data"):
            img1 = cv2.imread(im1, cv2.IMREAD_GRAYSCALE)
            res = stylized(content=im2, style=im1)
            cv2.imwrite(
                f"{OUT_DIR}{ds}/{str(Path(im1).name)}",
                np.array([res, img1, img1]).transpose(1, 2, 0),
            )


if __name__ == "__main__":
    main()
