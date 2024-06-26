import os
import shutil
import subprocess
import tempfile
from glob import glob
from pathlib import Path

import cv2


def stylized(content, style, flag=cv2.IMREAD_GRAYSCALE):
    tmpdir = tempfile.TemporaryDirectory()
    cdir = Path(tmpdir.name, "content")
    sdir = Path(tmpdir.name, "style")
    odir = Path(tmpdir.name, "output")
    os.mkdir(cdir)
    os.mkdir(sdir)
    os.mkdir(odir)

    shutil.copy(content, cdir / Path(content).name)
    shutil.copy(style, sdir / Path(style).name)
    subprocess.check_call(
        f"python test.py --content_dir {cdir} --style_dir {sdir} --output {odir}",
        cwd="./StyTR-2",
    )
    res = cv2.imread(glob(str(odir / "*_stylized_*"))[0], flag)

    tmpdir.cleanup()

    return res
