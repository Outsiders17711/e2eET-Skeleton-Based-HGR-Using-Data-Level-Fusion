# e2eET Skeleton Based HGR Using Data-Level Fusion
# Dynamic Hand Gestures Classification: Live Stream Demo
# pyright: reportGeneralTypeIssues=false
# pyright: reportWildcardImportFromLibrary=false
# ---------------------------------------------------------

import os
import json
import cv2 as cv
import numpy as np
from fastai.vision.all import *
from datetime import datetime, timedelta

from _helperFunctions import multiDetailsParser, hgrLogger


# [init.args.deets.defaults]
# ---------------------------
class Arguments:
    def __init__(self):
        self.mv_orientations = ["custom", "top-down", "front-away"]
        self.itr_scl_sizes = "null"

        cfg = json.load(open("./allConfigs.jsonc"))
        self.images_directory = cfg["images_directory"]
        self.hgr_archive = cfg["hgr_archive"]
        self.hgr_log = cfg["hgr_log"]
        self.debug_mode = cfg["debug_mode"]
        self.cpu_mode = cfg["cpu_mode"]

        self.idle = False
        self.data_files = []


args = Arguments()
deets = multiDetailsParser()
defaults.device = torch.device("cpu" if args.cpu_mode else "cuda:0")
Path(args.hgr_archive).mkdir(exist_ok=True)


# [init.learner.object]
# ---------------------------
from _functionsClasses import attachMetrics, e2eTunerLossWrapper

if os.name == 'nt':  # For Windows OS
    pkl_file = "./.sources/[bf75]-7G-[cm_td_fa]-Windows.pkl"
else:  # For Linux OS
    pkl_file = "./.sources/[bf75]-7G-[cm_td_fa]-Linux.pkl"

attachMetrics(e2eTunerLossWrapper, args.mv_orientations)
learn = load_learner(fname=pkl_file, cpu=args.cpu_mode)
str_dls_vocab = " ".join([f"{i}.{v}" for i, v in enumerate(learn.dls.vocab)])
print(f"INFO: dls.vocab=[{str_dls_vocab}]")


# [display.inferences]
# ---------------------------
window_name = "gestureClassInference.py"
# cv.namedWindow(window_name, cv.WINDOW_NORMAL)
# cv.setWindowProperty(window_name, cv.WND_PROP_TOPMOST, 1)


def _display(text):
    _display = np.zeros((240, 720, 3))
    o_x, o_y = 25, 50

    for line in text.splitlines():
        x = o_x
        if "\t" in line:
            x += 50
            line = line.replace("\t", "")
        cv.putText(_display, line, (x, o_y), cv.FONT_HERSHEY_SIMPLEX, 0.625, (0, 255, 0), 1)
        o_y += 35

    cv.imshow(window_name, _display)
    cv.waitKey(100)


# [gesture.inference]
# ---------------------------
def _archive():
    for gs_tag in Path(args.images_directory).iterdir():
        gs_tag.replace(f"{args.hgr_archive}/{gs_tag.name}")


def _inference_time(gs_tag):
    gs_time = [int(gs_tag[9:15][i : i + 2]) for i in (0, 2, 4)]
    gs_time = timedelta(hours=gs_time[0], minutes=gs_time[1], seconds=gs_time[2])
    return str(datetime.now() - gs_time)[11:19]


def main():
    data_files = L(dict.fromkeys([f.parent for f in get_image_files(args.images_directory)]))

    if data_files:
        print(
            f"\n{'-'*50}\nINFO: Loading data files @{args.mv_orientations} mVOs::",
            *data_files,
            sep="\n  - ",
        ) if args.debug_mode else None
        args.idle = False

        data_dl = learn.dls.test_dl(data_files)
        preds, targs, decoded = learn.get_preds(dl=data_dl, with_decoded=True)
        decoded = np.array([i.numpy() for i in decoded]).tolist()
        aggregate = np.array([i.numpy() for i in preds]).sum(axis=0).argmax(axis=1).tolist()

        for idx, gs in enumerate(data_dl.items):
            text = f">HGR: Inferences @{gs.parent}/{gs.stem} ~ [latency={_inference_time(gs.stem)}] ~ [aggregate={learn.dls.vocab[aggregate[idx]]}({aggregate[idx]})]"

            for jdx, vo in enumerate([*args.mv_orientations, "ensemble-tuner"]):
                pred = decoded[jdx][idx]
                text += f"\n\t\t- {vo}:: {learn.dls.vocab[pred]}({pred})"

            # _display(text)
            hgrLogger(text, log=args.hgr_log)

        _archive()

    else:
        text = "INFO: Waiting for data files..."
        # _display(text)
        print(text) if not (args.idle) and args.debug_mode else None
        args.idle = True


# [>>>>>]____________________________________________________________
if __name__ == "__main__":
    _archive()
    print("INFO: Initialized <gestureClassInference.py> ...")

    try:
        while True:
            main()
    except KeyboardInterrupt:
        cv.destroyAllWindows()
        print("INFO: KeyboardInterrupt received. Exiting...")
