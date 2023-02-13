# e2eET Skeleton Based HGR Using Data-Level Fusion
# Dynamic Hand Gestures Classification: LMDHG Dataset
# pyright: reportGeneralTypeIssues=false
# pyright: reportWildcardImportFromLibrary=false
# pyright: reportOptionalMemberAccess=false
# -----------------------------------------------
import os
import sys
import time
from datetime import datetime
import json
import argparse
import shutil
import pickle
from pathlib import Path
from threading import Thread
from multiprocessing import Pool

import playsound
import numpy as np

import vispy.io
from vispy import app, scene
from vispy.util.event import Event
import matplotlib.colors as mcolors

from denoise_gesture_sequences import denoise_gesture_sequences


# [DESC]: creating, for each gesture sequence:
# ------  @vispy: (spatio)temporal image(s) from a list of specified view orientations


# [HGR FUNCTIONS]______________________________________________________________
# setting the view orientation and zoom scale
def _set_view(v_orientation: str, camera_center=None, camera_scale=None):
    global view

    camera_center = means[seq_idx] if cfg.s_fitting == "adaptive" else camera_center
    camera_scale = diffs[seq_idx] if cfg.s_fitting == "adaptive-mean" else camera_scale

    # [NOTE]: the higher the scale factor, the further away the camera is!
    if cfg.s_fitting in ["min-max"]:
        view.camera.scale_factor = cfg.sz_canvas
    elif cfg.s_fitting in ["mean", "adaptive"]:
        if cfg.n_joint_coordinates == 2:
            view.camera.scale_factor = cfg.sz_canvas / 2
        elif cfg.n_joint_coordinates == 3:
            view.camera.scale_factor = np.mean(means * 1.25).round(2)
    elif cfg.s_fitting in ["adaptive-mean"]:
        if cfg.n_joint_coordinates == 2:
            view.camera.scale_factor = cfg.sz_canvas / 2 if camera_scale is None else camera_scale
        elif cfg.n_joint_coordinates == 3:
            view.camera.scale_factor = 0.5 if camera_scale is None else camera_scale
    elif cfg.s_fitting in ["legacy"]:
        view.camera.scale_factor = 500 if cfg.n_joint_coordinates == 2 else 120  # @==3

    if cfg.s_fitting in ["min-max", "mean", "adaptive-mean"]:
        view.camera.center = [cfg.sz_canvas / 2] * 3
    elif cfg.s_fitting in ["adaptive"]:
        view.camera.center = abs_means if camera_center is None else camera_center
    elif cfg.s_fitting in ["legacy"]:
        view.camera.center = [330, 250, 0] if cfg.n_joint_coordinates == 2 else [105, 30, 0]  # @==3

    if v_orientation == "top-down":  # TOP--> LOOKING DOWN
        view.camera.elevation, view.camera.azimuth = 0.0, 0.0
    elif v_orientation == "front-to":  # FRONT--> LOOKING @PARTICIPANT
        view.camera.elevation, view.camera.azimuth = -90.0, -180.0
    elif v_orientation == "front-away":  # FRONT--> LOOKING @OBSERVER
        view.camera.elevation, view.camera.azimuth = 90.0, 0.0
    elif v_orientation == "side-right":  # SIDE--> RIGHT
        view.camera.elevation, view.camera.azimuth = 0.0, 90.0
    elif v_orientation == "side-left":  # SIDE--> LEFT
        view.camera.elevation, view.camera.azimuth = 0.0, -90.0
    elif v_orientation == "custom":
        view.camera.scale_factor = 1000 if cfg.s_fitting == "min-max" else 700
        view.camera.elevation, view.camera.azimuth = -15.0, -135.0  # 30.0, 150.0 | -45.0, -45.0

    if not cfg.dataset_create_mode:
        il = (cfg.sz_canvas // 2) if cfg.s_fitting in ["min-max", "mean", "adaptive-mean"] else 0
        org, fl = ([il] * 3), (cfg.sz_canvas * 2)

        axis = scene.visuals.XYZAxis(parent=view.scene)  # x=red | y=green | z=blue
        axis_coords = np.array([org, [fl, il, il], org, [il, fl, il], org, [il, il, fl]])
        axis.set_data(width=1, pos=axis_coords)


# setting the size of the vispy canvas
def _set_canvas(sz_canvas):
    global canvas, info

    if cfg.s_fitting in ["min-max", "mean", "adaptive-mean"]:
        canvas.size = (sz_canvas, sz_canvas)
    elif cfg.s_fitting in ["adaptive"]:
        canvas.size = (sz_canvas * 1.5, sz_canvas)  # OR (sz_canvas * 16 / 9, sz_canvas)
    elif cfg.s_fitting in ["legacy"]:
        canvas.size = (1280, 720)

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

    # [NOTE] alter aplha channel gradations here
    colormap[:, 3] = np.linspace(0.1, 1, N) if cfg.vo_temporal_gradations else np.linspace(1, 1, N)

    return colormap


# giving each finger a distinct color
def _make_colored_fingers():
    global colors

    # collect the rightmost column of the nodes in the connection map
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
    global seq_idx, skel_idx, tips_coords, save_sequence, info

    while pause:
        time.sleep(0.1)
        return

    if skel_idx == len(ds_g_sequences[seq_idx]):
        skel_idx = 0
        if save_sequence or cfg.dataset_create_mode:
            _create_sequence_png(cfg.imgs_ds_dir)
            save_sequence = False

        seq_idx += 1
        if seq_idx == len(ds_g_sequences):
            seq_idx = 0
            if not cfg.dataset_create_mode:
                print(">>> all dataset gesture sequences parsed!")

            app.quit()

        del tips_coords
        tips_coords = {tip: [] for tip in cfg.finger_tips}

        if cfg.s_fitting in ["adaptive", "adaptive-mean"]:
            _set_view(tmp_v_orientation)

        if not cfg.dataset_create_mode and verbose:
            print(f"\t -> gesture {seq_idx+stt_lim}: {str_labels[seq_idx]}")

    skeleton_coords = ds_g_sequences[seq_idx][skel_idx]
    _coords_ = []

    for conn_pts in cfg.connection_map:
        pt1, pt2 = conn_pts
        _coords_.append(skeleton_coords[pt1])
        _coords_.append(skeleton_coords[pt2])

    if skel_idx % 1 == 0:  # set sampling here
        for tip in cfg.finger_tips:
            if sum(skeleton_coords[tip]) != 0:
                tips_coords[tip].append(skeleton_coords[tip])

    # [NOTE] show/hide skeletons in this block
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

    if not cfg.dataset_create_mode:
        info.text = f"i{seq_idx+stt_lim}.{str_labels[seq_idx]}\n[{tmp_v_orientation}] view\n{cfg.s_fitting} fitting"

    skel_idx += 1


def run_threaded_update(n):
    for i in range(n):
        ev = Event("dummy_event")
        update(ev)


def run_app(app):
    if sys.flags.interactive != 1:
        app.run()


def _create_sequence_png(dir):
    global pause
    fname = str_labels[seq_idx]
    prefix = f"{fname.split('-', maxsplit=1)[-1]}-"

    if cfg.dataset_create_mode:
        i_subset, i_gesture, i_id = fname.split("-")
        dir = dir.joinpath(f"{i_subset}/{i_gesture}/{i_id}")
        prefix = ""
    dir.mkdir(exist_ok=True, parents=True)

    pause = not pause
    for v in cfg.v_orientation:
        _set_view(v)
        img = canvas.render()
        vispy.io.write_png(f"{dir}/{prefix}{v}.png", img)
    pause = not pause

    _set_view(tmp_v_orientation)
    if not cfg.dataset_create_mode or verbose:
        print(f"> sequence <{seq_idx+stt_lim}> saved to <{dir}\\{prefix}*.png>")


def get_camera_details():
    print(
        f">>> @i{seq_idx+stt_lim}.{str_labels[seq_idx]}:",
        f"elevation {view.camera.elevation}",
        f"| azimuth {view.camera.azimuth}",
        f"| scale {round(view.camera.scale_factor, 2)}",
        f"| center {np.array(view.camera.center).astype(int)}",
    )


# [INIT VISPY SCENE OBJECTS]___________________________________________________
# build canvas, add viewbox and initialize visuals
canvas = scene.SceneCanvas(
    keys="interactive",
    title="create-imgs-LMDHDd",
    app="PyQt5",
    always_on_top=False,
    vsync=False,
    bgcolor="black",
    decorate=True,  # also sets `resizable=False`
)

view = canvas.central_widget.add_view()
view.camera = "turntable"  # "panzoom"
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
    global seq_idx, skel_idx, pause, tips_coords, save_sequence, canvas, tmp_v_orientation

    if event.key == "Escape" and (not cfg.dataset_create_mode):
        print(">>> exiting visualization GUI normally...")
        app.quit()
        return

    elif event.key in ["Left", "Right"] and (not cfg.dataset_create_mode):
        if event.key == "Left":
            tmp_v_orientation = cfg.allVOs[(cfg.allVOs.index(tmp_v_orientation) - 1) % len(cfg.allVOs)]
            _set_view(tmp_v_orientation)
        elif event.key == "Right":
            tmp_v_orientation = cfg.allVOs[(cfg.allVOs.index(tmp_v_orientation) + 1) % len(cfg.allVOs)]
            _set_view(tmp_v_orientation)

        info.text = f"i{seq_idx+stt_lim}.{str_labels[seq_idx]}\n[{tmp_v_orientation}] view\n{cfg.s_fitting} fitting"
        return

    elif event.key == " " and (not cfg.dataset_create_mode):
        print(">>> visualization paused..." if not pause else ">>> visualization resumed...")
        pause = not pause
        return

    elif event.key == "S" and (not cfg.dataset_create_mode):
        save_sequence = True
        return

    elif event.key == "C" and (not cfg.dataset_create_mode):
        get_camera_details()
        return

    elif event.key in ["P", "N", "Up", "Down"] and (not cfg.dataset_create_mode):
        skel_idx = 0

        # skipping to the [P]revious gesture sequence in the dataset
        if event.key in ["P", "Down"]:
            if seq_idx == 0:
                print(">>> first gesture sequence reached...")
            else:
                seq_idx -= 1

        # skipping to the [N]ext gesture sequence in the dataset
        elif event.key in ["N", "Up"]:
            if seq_idx == len(ds_g_sequences):
                print(">>> last gesture sequence reached...")
            else:
                seq_idx += 1

        del tips_coords
        tips_coords = {tip: [] for tip in cfg.finger_tips}
        print(f"\t -> gesture {seq_idx+stt_lim}: {str_labels[seq_idx]}") if verbose else None

    else:
        return


class loadConfigArguments:
    def __init__(self, args) -> None:
        config = json.load(open(args.config))
        self.allVOs = ["top-down", "front-to", "front-away", "side-right", "side-left", "custom"]

        self.connection_map = config["connection_map"]
        self.finger_tips = config["finger_tips"]
        self.fingers_colors = dict(config["fingers_colors"])
        self.n_joint_coordinates = config["n_joint_coordinates"]
        self.temporal_trails = config["temporal_trails"]

        self.create_test_subset = config["create_test_subset"]
        self.dataset_create_mode = config["dataset_create_mode"]
        self.vo_temporal_gradations = config["add_vo_temporal_gradations"]
        self.vo_skeletons = config["add_vo_skeletons"]

        self.w_visuals = config.setdefault("w_visuals", 3.5)
        self.sz_canvas = config.setdefault("sz_canvas", 960)
        self.n_lims = config.setdefault("n_processes", 1)
        self.fps = config.setdefault("fps", 10000)

        self.dataset_subset = config.setdefault("dataset_subset", "all")
        self.v_orientation = config.setdefault("view_orientation", "allVOs")
        self.s_fitting = config.setdefault("sequence_fitting", "adaptive-mean")
        self.denoise_dataset = config.setdefault("denoise_dataset", False)

        self.ds_pckl_file = config["dataset_pickle_file"].replace("<ND>", f"{self.n_joint_coordinates}d")

        self.imgs_ds_dir = Path(
            config["images_dataset_directory"]
            .replace("<ND>", f"{self.n_joint_coordinates}d")
            .replace("<NPX>", f"{self.sz_canvas}px")
            + f"-[{self.str_v_orientation()}].{self.s_fitting}"
        )

        if self.denoise_dataset:
            self.sz_denoising_filter = config["sz_denoising_filter"]
            self.n_denoised_skeletons = config["n_denoised_skeletons"]

        if self.dataset_create_mode:
            self.dataset_subset = "all"
            self.fps = np.inf
            self.n_lims = int(os.cpu_count()) - 2

        if self.v_orientation == "allVOs":
            self.v_orientation = self.allVOs

        if type(self.v_orientation) is str:
            self.v_orientation = [self.v_orientation]

        if self.n_joint_coordinates == 2:
            self.v_orientation = ["top-down"]

    # ---
    def str_v_orientation(self, delimiter="."):
        if type(self.v_orientation) == str:
            return self.v_orientation
        else:
            return f"{delimiter}".join(self.v_orientation).replace("-", "")

    # ---
    def init_gesture_sequences_and_labels(self, data):
        if self.dataset_subset == "train":
            self.ds_g_sequences = data["X_train"]
            str_labels = "train-" + np.array(data["train_details"], dtype=object)

        elif self.dataset_subset == "valid":
            self.ds_g_sequences = data["X_valid"]
            str_labels = "valid-" + np.array(data["valid_details"], dtype=object)

        else:  # elif self.dataset_subset == "all":
            self.ds_g_sequences = np.concatenate((data["X_train"], data["X_valid"]), axis=0)
            str_labels = np.concatenate(
                (
                    "train-" + np.array(data["train_details"], dtype=object),
                    "valid-" + np.array(data["valid_details"], dtype=object),
                )
            )

        # self.ds_g_sequences = -(self.ds_g_sequences)
        l_dataset, n_skeletons, l_coords = self.ds_g_sequences.shape
        self.ds_g_sequences = self.ds_g_sequences.reshape(
            (l_dataset, n_skeletons, -1, self.n_joint_coordinates)
        )

        if self.denoise_dataset:
            self.imgs_ds_dir = Path(
                str(self.imgs_ds_dir).replace("<DNSD?>", f"denoised({self.sz_denoising_filter})")
            )
            self.ds_g_sequences = denoise_gesture_sequences(
                self.ds_g_sequences,
                sz_filter=self.sz_denoising_filter,
                n_samples=self.n_denoised_skeletons,
            )

        else:
            self.imgs_ds_dir = Path(str(self.imgs_ds_dir).replace("<DNSD?>", "noisy(raw)"))

        return l_dataset, n_skeletons, str_labels

    # ---
    def get_class_distribution(self, labels):
        classDistribution = {}
        for _lbl in labels:
            _lbl = _lbl.split("-")[1]
            if _lbl not in classDistribution:
                classDistribution[_lbl] = 1
            else:
                classDistribution[_lbl] += 1

        print(f">>> {classDistribution = } [{len(classDistribution)}]")

    # ---
    def do_sequence_fitting(self):
        a, b, c, d = self.ds_g_sequences.shape
        abs_means, means, diffs = [], [], []

        n_hands = np.sum(np.sum(self.ds_g_sequences[:, (b // 2), :23], axis=2), axis=1)
        self.n_hands = (n_hands != 0.0).astype(int) + 1
        adj_scaling = ((n_hands == 0.0).astype(int) * 0.5) + 1  # [NOTE] when one hand is visible

        if cfg.s_fitting == "min-max":
            padding = 30
            isz, fsz = padding, cfg.sz_canvas - padding
            mins = np.min(np.min(self.ds_g_sequences, axis=2), axis=1).reshape((a, 1, 1, d))
            maxes = np.max(np.max(self.ds_g_sequences, axis=2), axis=1).reshape((a, 1, 1, d))
            self.ds_g_sequences = (fsz - isz) * ((self.ds_g_sequences - mins) / (maxes - mins)) + isz

        elif cfg.s_fitting == "mean":
            means = np.mean(np.mean(self.ds_g_sequences, axis=2), axis=1).reshape((a, 1, 1, d))
            self.ds_g_sequences -= means - cfg.sz_canvas / 2

        elif cfg.s_fitting == "adaptive":
            means = np.mean(np.mean(self.ds_g_sequences, axis=2), axis=1)
            abs_means = np.mean(means, axis=0).round(4)

        elif cfg.s_fitting == "adaptive-mean":
            mins = np.min(np.min(np.min(self.ds_g_sequences, axis=3), axis=2), axis=1)
            maxes = np.max(np.max(np.max(self.ds_g_sequences, axis=3), axis=2), axis=1)
            means = np.mean(np.mean(self.ds_g_sequences, axis=2), axis=1).reshape((a, 1, 1, d))
            self.ds_g_sequences -= means - cfg.sz_canvas / 2
            if cfg.n_joint_coordinates == 2:
                diffs = (maxes - mins + (0.25 * cfg.sz_canvas)).astype(int)  # add a little "padding"
            elif cfg.n_joint_coordinates == 3:
                diffs = (maxes - mins + 0.125).round(2)  # add a little "padding"

        elif cfg.s_fitting == "legacy" and cfg.n_joint_coordinates == 3:
            self.ds_g_sequences *= 250.0
            self.ds_g_sequences[..., 1] -= 50.0

        return self.ds_g_sequences, abs_means, means, (diffs * adj_scaling)

    # ---
    def do_train_valid_test_split(self):
        rng = np.random.default_rng(17711)

        ds_valid = Path(f"{self.imgs_ds_dir}/valid/").rglob("*.png")
        ds_valid = np.array(list(dict.fromkeys([f.parent for f in ds_valid])), dtype=object)
        rng.shuffle(ds_valid)

        ds_test = ds_valid[: len(ds_valid) // 2].tolist()
        for s in ds_test:
            s_test = list(s.parts)
            s_test[-3] = "test"
            s_test = "/".join(s_test)
            shutil.move(s, s_test)


def main(stt_end_args):
    global ds_g_sequences, str_labels, means, diffs, stt_lim, canvas
    (stt_lim, end_lim), n = stt_end_args

    ds_g_sequences = ds_g_sequences[stt_lim:end_lim]
    str_labels = str_labels[stt_lim:end_lim]
    means = means[stt_lim:end_lim] if cfg.s_fitting in ["mean", "adaptive", "adaptive-mean"] else []
    diffs = diffs[stt_lim:end_lim] if cfg.s_fitting in ["adaptive-mean"] else []

    if not cfg.dataset_create_mode and verbose:
        print(f"\t@pool {n:02}: {stt_lim=:04} -> {end_lim=:04} == {end_lim-stt_lim} sequences")

    _set_canvas(cfg.sz_canvas)
    _set_view(tmp_v_orientation)
    _make_colored_fingers()
    timer = app.Timer(interval=1.0 / cfg.fps, connect=update, start=True)

    if cfg.dataset_create_mode:
        thread = Thread(target=run_app, args=(app,))
        thread.start()
        run_threaded_update((end_lim - stt_lim + 1) * n_skeletons)
        thread.join()
        return None
    else:
        canvas.position = window_positions[n % len(window_positions)]
        canvas.show(visible=True)
        run_app(app)
        return get_camera_details()


# [LOAD CONFIG JSON & DATASET PCLK]____________________________________________
start_timer = datetime.now()
ap = argparse.ArgumentParser()
ap.add_argument(
    "-c",
    "--config",
    required=False,
    default="modules/.configs/lmdhg-v5-default.hgr-config",
    help="path to the JSON configuration file",
)
args = ap.parse_args()
cfg = loadConfigArguments(args)


# [>>>>>]____________________________________________________________
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
tmp_v_orientation = cfg.v_orientation[-1]

# init global variables
seq_idx = skel_idx = 0
pause = save_sequence = False
verbose = False
window_positions = np.array([[0, 0], [100, 25], [200, 50], [300, 75]], dtype=int) + 25

# load `dataset_pickle_file`
pckl_file = open(cfg.ds_pckl_file, "rb")
data = pickle.load(pckl_file, encoding="latin1")
pckl_file.close()

# init gesture sequences and labels
l_dataset, n_skeletons, str_labels = cfg.init_gesture_sequences_and_labels(data)

# fit gesture sequences to scene
ds_g_sequences, abs_means, means, diffs = cfg.do_sequence_fitting()


# [>>>>>]____________________________________________________________
if __name__ == "__main__":
    print("-" * 50)
    time_elapsed = str(datetime.now() - start_timer)[2:-7]
    print(f">>> @{time_elapsed}m; dataset and config files loaded successfully...")
    print(f">>> plotting <{ds_g_sequences.shape= }> dataset gesture sequences...")
    cfg.get_class_distribution(str_labels)
    print(f">>> viewOrientations = {cfg.v_orientation} [{len(cfg.v_orientation)}]")
    print("-" * 50)

    if cfg.imgs_ds_dir.exists() and cfg.dataset_create_mode:
        print(f"<<< Warning: `images_dataset_directory` @{cfg.imgs_ds_dir} exists! Deleting...")
        shutil.rmtree(cfg.imgs_ds_dir, ignore_errors=True)

    start_timer = datetime.now()
    lims = np.linspace(0, l_dataset, cfg.n_lims + 1, dtype=int)
    starts_ends = [list(lims[i : i + 2]) for i in range(cfg.n_lims)]

    with Pool(cfg.n_lims) as pool:
        pool.map(main, zip(starts_ends, list(range(cfg.n_lims))))

    if cfg.dataset_create_mode and cfg.create_test_subset:
        cfg.do_train_valid_test_split()

    if cfg.dataset_create_mode:
        time_elapsed = str(datetime.now() - start_timer)[:-7]
        print(f">>> @0{time_elapsed}h; <{len(ds_g_sequences)=}> encoded dataset generated!")
        print(f">>> @{cfg.imgs_ds_dir}")
        playsound.playsound(r"./modules/notification.mp3", block=True)

# [>>>>>]____________________________________________________________
# python modules/create_imgs_v5_LMDHGd_mVOs.py -c "modules/.configs/lmdhg-v5-default.hgr-config"
# mVOs: 00:06:42h
