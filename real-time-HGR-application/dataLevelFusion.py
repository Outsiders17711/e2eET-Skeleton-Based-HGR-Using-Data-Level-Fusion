# e2eET Skeleton Based HGR Using Data-Level Fusion
# Dynamic Hand Gestures Classification: Live Stream Demo
# pyright: reportGeneralTypeIssues=false
# pyright: reportWildcardImportFromLibrary=false
# pyright: reportOptionalMemberAccess=false
# -----------------------------------------------

import sys
import time
import json
from pathlib import Path
from collections import deque

import vispy.io
from vispy import app, scene
from vispy.util.event import Event

import numpy as np
from scipy import ndimage
import matplotlib.colors as mcolors

from _helperFunctions import hgrLogger


# [HGR FUNCTIONS]______________________________________________________________
# setting the view orientation and zoom scale
def _set_view(v_orientation: str):
    global view

    view.camera.scale_factor = cfg.cam_scaling
    view.camera.center = [cfg.sz_canvas / 2] * 3

    if v_orientation == "top-down":  # TOP--> LOOKING DOWN
        view.camera.elevation, view.camera.azimuth = 0.0, 0.0
    elif v_orientation == "front-to":  # FRONT--> LOOKING @PARTICIPANT
        view.camera.elevation, view.camera.azimuth = 90.0, 180.0
    elif v_orientation == "front-away":  # FRONT--> LOOKING @OBSERVER
        view.camera.elevation, view.camera.azimuth = -90.0, 0.0
    elif v_orientation == "side-right":  # SIDE--> RIGHT
        view.camera.elevation, view.camera.azimuth = 0.0, -90.0
    elif v_orientation == "side-left":  # SIDE--> LEFT
        view.camera.elevation, view.camera.azimuth = 0.0, 90.0
    elif v_orientation == "custom":
        view.camera.scale_factor = 0.85
        view.camera.elevation, view.camera.azimuth = 30.0, -132.5

    if cfg.debug_mode:
        il = cfg.sz_canvas // 2
        org, fl = ([il] * 3), (cfg.sz_canvas * 2)

        axis = scene.visuals.XYZAxis(parent=view.scene)  # x=red | y=green | z=blue
        axis_coords = np.array([org, [fl, il, il], org, [il, fl, il], org, [il, il, fl]])
        axis.set_data(width=1, pos=axis_coords)


# setting the size of the vispy canvas
def _set_canvas(sz_canvas):
    global canvas, info

    canvas.size = (sz_canvas, sz_canvas)
    info.pos = [canvas.size[0] - 10, 10, 0]

    canvas.show(visible=True)
    canvas.show(visible=False)


# getting the mapping of finger_tips colors
def _get_tip_colormap(tip, N):
    global cfg

    _tipcolor = cfg.fingers_colors[tip]
    _tipcolor = mcolors.to_rgb(mcolors.CSS4_COLORS[_tipcolor])

    colormap = np.ones((N, 4))
    colormap[:, 0:3] = _tipcolor
    colormap[:, 3] = np.linspace(0, 1, N) if cfg.vo_temporal_gradations else np.linspace(1, 1, N)

    return colormap


# giving each finger a distinct color
def _make_colored_fingers():
    global colors

    right_cM = [item[1] for item in cfg.connection_map]
    offset = 0

    for node in range(len(cfg.fingers_colors)):
        _nodecolor = cfg.fingers_colors[node]
        _nodecolor = mcolors.to_rgb(mcolors.CSS4_COLORS[_nodecolor])

        for right_idx, right_node in enumerate(right_cM):
            if right_node == node:
                colors[right_idx * 2 + offset, :-1] = _nodecolor
                colors[right_idx * 2 + offset + 1, :-1] = _nodecolor


# [VISPY FUNCTIONS]____________________________________________________________
def update(null):
    global viz_frozen, idx_sleleton, tips_coords, info, l_data_files

    while viz_frozen:
        time.sleep(0.1)
        return

    # if (idx_sleleton == (cfg.gs_length - 1)) and not (l_data_files): viz_frozen = not (viz_frozen)
    # if idx_sleleton == (cfg.gs_length - 1): viz_frozen = not (viz_frozen)

    if idx_sleleton == cfg.gs_length:
        idx_sleleton = 0

        if l_data_files:
            create_sequence_png()
            l_data_files -= 1

        del tips_coords
        tips_coords = {tip: [] for tip in cfg.finger_tips}

        main()
        _set_view(viz_VO)

    skeleton_coords = cfg.gs_data[idx_sleleton]
    _coords_ = []

    for conn_pts in cfg.connection_map:
        pt1, pt2 = conn_pts
        _coords_.append(skeleton_coords[pt1])
        _coords_.append(skeleton_coords[pt2])

    if idx_sleleton % 1 == 0:  # set sampling here
        for tip in cfg.finger_tips:
            if sum(skeleton_coords[tip]) != 0:
                tips_coords[tip].append(skeleton_coords[tip])

    if cfg.vo_skeletons:
        _coords_ = np.array(_coords_)
        skeleton.set_data(_coords_, color=colors, width=cfg.w_visuals, connect="segments")
        skeleton_nodes.set_data(_coords_, size=2 * cfg.w_visuals, face_color=colors, edge_color=colors)

    for tip in cfg.finger_tips:
        if not tips_coords[tip]:  # catch possible errors
            continue

        color = _get_tip_colormap(tip, len(tips_coords[tip]))
        _tips_ = np.array(tips_coords[tip])
        if cfg.temporal_trails == "lines":
            tips_visuals[tip].set_data(_tips_, color=color, width=cfg.w_visuals)
        elif cfg.temporal_trails == "markers":
            tips_visuals[tip].set_data(_tips_, face_color=color, edge_color=color, size=cfg.w_visuals)

    if cfg.debug_mode and l_data_files:
        info.text = f"@{cfg.gs_tag.name}\n[{cfg.str_v_orientation()}*] mVOs"

    idx_sleleton += 1


def run_threaded_update(n):
    for i in range(n):
        ev = Event("dummy_event")
        update(ev)


def run_app(app):
    if sys.flags.interactive != 1:
        app.run()


def create_sequence_png():
    global viz_frozen
    viz_frozen = not viz_frozen

    for v in cfg.v_orientation:
        _set_view(v)
        img = canvas.render()
        png = f"{cfg.gs_data_directory}/{cfg.gs_tag.name}/{v}.png"
        vispy.io.write_png(png, img)

    viz_frozen = not viz_frozen
    cfg.gs_tag.replace(f"{cfg.gs_images_directory}/{cfg.gs_tag.name}")

    hgrLogger(f">HGR: @{cfg.gs_tag.name} gesture sequence visualized and saved.", log=cfg.hgr_log)
    _set_view(viz_VO)


def get_camera_details():
    print(
        f"@get_camera_details():",
        f"elevation {view.camera.elevation}",
        f"| azimuth {view.camera.azimuth}",
        f"| scale {round(view.camera.scale_factor, 2)}",
        f"| center {np.array(view.camera.center).astype(int)}",
    )


# [INIT VISPY SCENE OBJECTS]___________________________________________________

# build canvas, add viewbox and initialize visuals
canvas = scene.SceneCanvas(
    keys="interactive",
    title="e2eET HGR: Vispy Data-Level Fusion",
    app="PyQt5",
    always_on_top=True,
    vsync=False,
    bgcolor="black",
    decorate=True,  # also sets `resizable=False`
    position=(1275, 40),
)

view = canvas.central_widget.add_view()
view.camera = "turntable"
view.camera.fov, view.camera.roll, view.camera.distance = 0.0, 0.0, 0.0

skeleton = scene.visuals.Line()
view.add(skeleton)
skeleton_nodes = scene.visuals.Markers()
view.add(skeleton_nodes)
info = scene.Text(
    text="",
    parent=canvas.scene,
    color="darkorange",
    bold=True,
    font_size=12.0,
    anchor_x="right",
    anchor_y="bottom",
    face="Calibri",
)


@canvas.events.key_press.connect
def on_key_press(event):
    global viz_frozen, viz_VO

    if cfg.debug_mode:
        if event.key == "Escape":
            print("INFO: Exiting visualization GUI normally.")
            app.quit()
            return

        elif event.key in ["Left", "Right"]:
            if event.key == "Left":
                viz_VO = cfg.allVOs[(cfg.allVOs.index(viz_VO) - 1) % len(cfg.allVOs)]
                _set_view(viz_VO)
            elif event.key == "Right":
                viz_VO = cfg.allVOs[(cfg.allVOs.index(viz_VO) + 1) % len(cfg.allVOs)]
                _set_view(viz_VO)

            if cfg.debug_mode and l_data_files:
                info.text = f"@{cfg.gs_tag.name}\n[{cfg.str_v_orientation()}*] mVOs"

            return

        elif event.key == " ":
            print("INFO: Visualization paused." if not viz_frozen else "INFO: Visualization resumed.")
            viz_frozen = not viz_frozen
            return

        elif event.key == "C":
            get_camera_details()
            return

        else:
            return

    else:
        return


class loadConfigArguments:
    def __init__(self) -> None:
        config = json.load(open("./allConfigs.jsonc"))
        self.allVOs = ["top-down", "front-to", "front-away", "side-right", "side-left", "custom"]

        self.dhg1428_mode = config["dhg1428_mode"]
        prefix = "dhg" if self.dhg1428_mode else "mp"
        self.connection_map = config[f"{prefix}_connection_map"]
        self.finger_tips = config[f"{prefix}_finger_tips"]
        self.fingers_colors = dict(config[f"{prefix}_fingers_colors"])

        self.debug_mode = config["debug_mode"]
        self.vo_temporal_gradations = config["add_vo_temporal_gradations"]
        self.vo_skeletons = config["add_vo_skeletons"]
        self.temporal_trails = config["temporal_trails"]

        self.w_visuals = config.setdefault("w_visuals", 3.5)
        self.sz_canvas = config.setdefault("sz_canvas", 960)
        self.n_lims = config.setdefault("n_processes", 1)
        self.fps = config.setdefault("fps", np.inf)

        self.hgr_log = Path(config["hgr_log"])
        self.gs_images_directory = Path(config["images_directory"])
        self.gs_data_directory = Path(config["data_directory"])
        for gs_tag in self.gs_data_directory.glob("*"):
            # gs_tag.replace(f"{self.gs_images_directory}/{gs_tag.name}")
            pass

        self.v_orientation = config.setdefault("view_orientation", ["front-to"])
        if self.v_orientation == "allVOs":
            self.v_orientation = self.allVOs
        if type(self.v_orientation) is str:
            self.v_orientation = [self.v_orientation]

    # ---
    def str_v_orientation(self, delimiter="."):
        if type(self.v_orientation) == str:
            return self.v_orientation
        else:
            return f"{delimiter}".join(self.v_orientation).replace("-", "")

    # ---
    def init_gesture_sequence(self, gs_tag=Path("./")):
        self.gs_tag = gs_tag
        self.gs_data = np.load(f"{gs_tag}/gs_sequence.npy", allow_pickle=True)

        # --- **tweak here** ---
        # self.gs_data = np.abs(self.gs_data)

        self.gs_length = len(self.gs_data) * 2
        self.transform_mediapipe_to_DHG1428() if cfg.dhg1428_mode else None
        self.interpolate_gesture_sequence(self.gs_length)

        mins = np.min(np.min(np.min(self.gs_data, axis=2), axis=1), axis=0)
        maxes = np.max(np.max(np.max(self.gs_data, axis=2), axis=1), axis=0)
        means = np.mean(np.mean(self.gs_data, axis=1), axis=0)

        self.gs_data -= means - self.sz_canvas / 2.0
        cam_padding = (-0.5) if cfg.dhg1428_mode else (0.125)
        self.cam_scaling = (maxes - mins + (cam_padding)).round(2)

    # ---
    def transform_mediapipe_to_DHG1428(self):
        # mirror the skeleton along the x axis; this shrinks the visualization, compensate `cam_scaling` with `cam_padding`
        self.gs_data *= [-1, 1, 1]

        # estimate the palm center as the midpoint of (the mean of the finger bases) and (the thumb base)
        mp_finger_bases = [5, 9, 13, 17]
        finger_bases_centers = np.mean(self.gs_data[:, mp_finger_bases], axis=1)
        thumb_bases = self.gs_data[:, 0]
        palm_centers = np.mean([finger_bases_centers, thumb_bases], axis=0)

        # insert the palm center into the same sequence index as DHG1428
        self.gs_data = np.insert(self.gs_data, 1, palm_centers, axis=1)

    # ---
    def interpolate_gesture_sequence(self, target_length=250):
        # flatten the joint coordinates in the sequence
        n_skeletons, n_joints, n_coords = self.gs_data.shape
        gs = self.gs_data.reshape((n_skeletons, -1))

        # interpolate sequence to target length
        _gs = []
        for _skeleton in range(np.size(gs, 1)):
            _gs.append(ndimage.zoom(gs.T[_skeleton], target_length / len(gs), mode="reflect"))

        # reform the joint coordinates in the sequence
        self.gs_data = np.array(_gs).T.reshape((-1, n_joints, n_coords))


# [>>>>>]____________________________________________________________
cfg = loadConfigArguments()

# finger tips history initializations
tips_coords = {tip: [] for tip in cfg.finger_tips}

if cfg.temporal_trails == "lines":
    tips_visuals = {tip: scene.visuals.Line() for tip in cfg.finger_tips}
else:
    tips_visuals = {tip: scene.visuals.Markers() for tip in cfg.finger_tips}

for visual in tips_visuals.values():
    view.add(visual)

# setting hands colors and transparencies
N = len(cfg.connection_map) * 2
colors = np.ones((N, 4), dtype=np.float32)
viz_VO = cfg.v_orientation[-1]

# init global variables
idx_sleleton = 0
viz_frozen = viz_idle = False
data_files = deque()
l_data_files = 0


# [>>>>>]____________________________________________________________
# init, interpolate and fit gesture sequences
def main():
    global data_files, l_data_files, viz_idle

    if l_data_files:
        cfg.init_gesture_sequence(data_files.pop())

    else:
        data_files = deque(sorted(cfg.gs_data_directory.iterdir(), reverse=True))
        l_data_files = len(data_files)

        if l_data_files:
            print(
                f"\n{'-'*50}\nINFO: Loading data files @[{cfg.str_v_orientation()}] mVOs::",
                *data_files,
                sep="\n  - ",
            ) if cfg.debug_mode else None
            main()
            viz_idle = False

        else:
            if not (viz_idle) and cfg.debug_mode:
                print("INFO: Waiting for data files...")
                info.text = "INFO: Waiting for data files..."

            viz_idle = True
            cfg.init_gesture_sequence()


# [>>>>>]____________________________________________________________
if __name__ == "__main__":
    main()
    _set_canvas(cfg.sz_canvas)
    _set_view(viz_VO)
    _make_colored_fingers()
    print("INFO: Initialized <dataLevelFusion.py> ...")

    timer = app.Timer(interval=1.0 / cfg.fps, connect=update, start=True)
    canvas.show(visible=True)

    run_app(app)
    get_camera_details()
