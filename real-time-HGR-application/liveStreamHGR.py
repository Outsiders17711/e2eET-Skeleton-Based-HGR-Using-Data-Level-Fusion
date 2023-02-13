# e2eET Skeleton Based HGR Using Data-Level Fusion
# Dynamic Hand Gestures Classification: Live Stream Demo
# pyright: reportGeneralTypeIssues=false
# pyright: reportWildcardImportFromLibrary=false
# pyright: reportOptionalMemberAccess=false
# -----------------------------------------------

import re
import json
import time
import subprocess
from pathlib import Path
from collections import deque
from datetime import datetime
from itertools import starmap, repeat

import cv2 as cv
import numpy as np
import matplotlib.colors as mc
from _mediapipePoseEstimation import HandDetector
from _helperFunctions import hgrLogger


# [GLOBALS]__________________________________________________________
cfg = json.load(open("./allConfigs.jsonc"))
# ---
gs_deque = deque(maxlen=cfg["MAX_HISTORY"])
gs_length = cfg["MAX_HISTORY"] - cfg["HISTORY_BUFFER"]
gs_minimum = int(cfg["LOGGER_THRESHOLD"] * cfg["MAX_HISTORY"])
hgr_log = cfg["hgr_log"]
# ---
str_colors = dict(cfg["mp_fingers_colors"])
connection_map = cfg["mp_connection_map"]
finger_tips = cfg["mp_finger_tips"]
mp_drawings = cfg["MP_DRAWINGS"]

rgb_colors = []

# [FUNCTIONS]________________________________________________________
def _color_fingers():
    global rgb_colors

    nodes = str_colors.keys()
    rgb_colors = [mc.to_rgb(mc.CSS4_COLORS[str_colors[n]])[::-1] for n in nodes]
    rgb_colors = (np.array(rgb_colors) * 255).astype(int).tolist()


def _draw_landmarks(img, lmCoords_2D):
    lmCoords_2D = lmCoords_2D[:, :-1].astype(int)

    for (node1, node2) in connection_map:
        cv.line(img, lmCoords_2D[node1], lmCoords_2D[node2], rgb_colors[node1], 2)

    for (node, coords) in enumerate(lmCoords_2D):
        if node in finger_tips:
            cv.circle(img, coords, 5, rgb_colors[node], cv.FILLED)
            cv.circle(img, coords, 3, (0, 0, 0), cv.FILLED)
        else:
            cv.circle(img, coords, 3, rgb_colors[node], cv.FILLED)

    return img


def gs_logger():
    global gs_deque

    gs_tag = re.sub("[-:]", "", str(datetime.now())).replace(" ", ".")
    hgrLogger(f"{'-'*25}\n>HGR: @{gs_tag}: {len(gs_deque)=:02}->>-", log=hgr_log, end="")

    gs_tag = Path(f"{cfg['data_directory']}/{gs_tag}")
    gs_tag.mkdir(parents=True, exist_ok=True)

    n_skeletons = min(len(gs_deque), gs_length)
    gs = np.array(list(starmap(gs_deque.popleft, repeat((), n_skeletons))))

    np.save(f"{gs_tag}/gs_sequence", gs)
    hgrLogger(f"{len(gs_deque):02} | {gs.shape=}", log=hgr_log)


def live_stream_hgr(nD):
    time.sleep(5)
    nD = nD.upper()
    assert nD in ["2D", "3D"], "ValueError@ nD parameter"

    # --- init camera window, set frame dimensions, and keep on top
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cfg["FRAME_SIZE"])
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, (cfg["FRAME_SIZE"] * 9 / 16))

    window_name = "e2eET HGR: Mediapipe Skeleton Estimation"
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.setWindowProperty(window_name, cv.WND_PROP_TOPMOST, 1)
    cv.moveWindow(window_name, x=50, y=15)

    # --- init mediapipe skeleton etimation module
    detector = HandDetector(detectionCon=0.85, maxHands=1)
    _color_fingers()
    print("INFO: Initialized <liveStreamHGR.py> ...")

    while True:
        success, img = cap.read()
        hand, img = detector.findHands(img, draw=mp_drawings)

        if hand:
            hand = hand[0]

            lmCoords = hand[f"lmCoords_{nD}"]
            gs_deque.append(lmCoords)
            print(lmCoords.tolist(), end="\n\n") if cfg["VERBOSE"] else None

            if not mp_drawings:
                img = _draw_landmarks(img, hand[f"lmCoords_2D"])

            # ---
            # save the deque if gs_length is reached
            # gs_logger() if (len(gs_deque) == cfg["MAX_HISTORY"]) else None

        else:
            # ---
            # if hand tracking is lost, save the deque if it is not empty
            # gs_logger() if len(gs_deque) > gs_minimum else None
            pass

        cv.imshow(window_name, cv.flip(img, 1))
        key = cv.waitKey(1)

        # release camera object and terminate live stream demo
        if key == 27:  # escape key
            cap.release()
            break

        # manual command to save the deque (all automatic saves disabled above)
        elif (key == 32) and (len(gs_deque) > gs_minimum):  # spacebar
            gs_logger()
            cv.waitKey(1500)

        # manual command to clear the deque (slight pause added for feedback)
        elif key == ord("c"):
            gs_deque.clear()
            # cv.waitKey(150)


# [>>>>>]____________________________________________________________
if __name__ == "__main__":
    gesture_inference = subprocess.Popen(args="python ./gestureClassInference.py")
    data_level_fusion = subprocess.Popen(args="python ./dataLevelFusion.py")
    vispy_output_gui = subprocess.Popen(args="python ./vispyOutputGUI.py")

    try:
        live_stream_hgr(nD="3d")
        print("\nINFO: Exiting liveStreamHGR normally. ", end="")

    except Exception as ex: print(f"\nERROR: >- {ex}. -< ", end="")

    # --- close all opencv/vispy windows and child processes
    finally:
        gesture_inference.terminate()
        data_level_fusion.terminate()
        vispy_output_gui.terminate()
        cv.destroyAllWindows()
        print("All child processes terminated!")
