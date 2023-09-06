# [TITLE]: Dynamic Human Action Classification: SBU Kinect Interaction Dataset
# [DESC]:  For each action sequence, create a vispy spatiotemporal image(s) from a list of specified mVOs
# pyright: reportGeneralTypeIssues=false
# pyright: reportWildcardImportFromLibrary=false
# pyright: reportOptionalMemberAccess=false
# -----------------------------------------------
import os
import sys
import time
import json
import shutil
import pickle
from PIL import Image
from pathlib import Path
from threading import Thread
from datetime import datetime
from multiprocessing import Pool

import vispy.io
from vispy import app, scene
from vispy.util.event import Event

import numpy as np
import matplotlib.colors as mcolors


# [HGR FUNCTIONS - ADAPTIVE MEAN]______________________________________________
# setting the view orientation and camera properties for adaptive-mean sequence fitting
def _set_view(v_orientation: str):
    global view

    view.camera.scale_factor = diffs[idx_sequence]
    view.camera.center = [cfg.sz_canvas / 2] * 3

    if v_orientation == "top-down":
        view.camera.elevation, view.camera.azimuth = 0.0, 0.0
    elif v_orientation == "front-to":
        view.camera.elevation, view.camera.azimuth = 90.0, 180.0
    elif v_orientation == "front-away":
        view.camera.elevation, view.camera.azimuth = -90.0, 0.0
    elif v_orientation == "side-right":
        view.camera.elevation, view.camera.azimuth = 0.0, -90.0
    elif v_orientation == "side-left":
        view.camera.elevation, view.camera.azimuth = 0.0, 90.0
    elif v_orientation == "custom":
        view.camera.elevation, view.camera.azimuth = 25.0, 115.0  # 30.0, -132.5

    if not (cfg.dataset_create_mode):
        il = cfg.sz_canvas // 2
        org, fl = ([il] * 3), (cfg.sz_canvas * 2)

        axis = scene.visuals.XYZAxis(parent=view.scene)  # x=red | y=green | z=blue
        axis_coords = np.array([org, [fl, il, il], org, [il, fl, il], org, [il, il, fl]])
        axis.set_data(width=1, pos=axis_coords)


# setting the size and visibiity of the vispy canvas
def _set_canvas(sz_canvas):
    global canvas, info

    canvas.size = (sz_canvas, sz_canvas)
    info.pos = [canvas.size[0] - 10, 10, 0]
    canvas.show(visible=True)
    canvas.show(visible=False)


# getting the mapping of skeleton_tips s_colors
def _get_tip_colormap(tip, N):
    global cfg

    _tipcolor = cfg.skeleton_colors[tip]
    _tipcolor = mcolors.to_rgb(mcolors.CSS4_COLORS[_tipcolor])
    colormap = np.ones((N, 4))
    colormap[:, 0:3] = _tipcolor
    colormap[:, 3] = np.linspace(0, 1, N) if cfg.vo_temporal_gradations else np.linspace(1, 1, N)

    return colormap


# setting the color for each skeleton joint
def _make_colored_fingers():
    global s_colors, h_colors

    right_cM = [item[1] for item in cfg.connection_map]
    offset = 0

    for node in range(len(cfg.skeleton_colors)):
        _nodecolor = cfg.skeleton_colors[node]
        _nodecolor = mcolors.to_rgb(mcolors.CSS4_COLORS[_nodecolor])

        for right_idx, right_node in enumerate(right_cM):
            if right_node == node:
                s_colors[right_idx * 2 + offset, :-1] = _nodecolor
                s_colors[right_idx * 2 + offset + 1, :-1] = _nodecolor

    h_colors = s_colors[[0, len(s_colors) // 2]]


# [VISPY FUNCTIONS]____________________________________________________________
def update(null):
    global idx_sequence, idx_skeleton, tips_coords, save_sequence, info

    while pause:
        time.sleep(0.1)
        return

    if idx_skeleton == len(ds_a_sequences[idx_sequence]):
        idx_skeleton = 0
        if save_sequence or cfg.dataset_create_mode:
            _create_sequence_png(cfg.imgs_ds_dir)
            save_sequence = False

        idx_sequence += 1
        if idx_sequence == len(ds_a_sequences):
            idx_sequence = 0
            print(">>> all dataset action sequences parsed!") if (not cfg.dataset_create_mode) else None
            app.quit()

        del tips_coords
        tips_coords = {tip: [] for tip in cfg.skeleton_tips}
        _set_view(canvasVO)

        if not cfg.dataset_create_mode and verbose:
            print(f"\t -> action {idx_sequence+stt_lim}: {ds_a_labels[idx_sequence]}")

    i_i_skeleton = ds_a_sequences[idx_sequence][idx_skeleton]
    s_coords = []

    for (pt1, pt2) in cfg.connection_map:
        s_coords.append(i_i_skeleton[pt1])
        s_coords.append(i_i_skeleton[pt2])

    if idx_skeleton % 1 == 0:  # set sampling here
        for tip in cfg.skeleton_tips:
            if sum(i_i_skeleton[tip]) != 0:
                tips_coords[tip].append(i_i_skeleton[tip])

    if cfg.vo_skeletons:
        s_coords = np.array(s_coords)
        h_coords = s_coords[[0, len(s_coords) // 2]]

        skeleton.set_data(s_coords, color=s_colors, width=cfg.w_visuals, connect="segments")
        joint_nodes.set_data(s_coords, size=2 * cfg.w_visuals, face_color=s_colors, edge_color=s_colors)
        head_nodes.set_data(h_coords, size=10 * cfg.w_visuals, face_color=h_colors, edge_color=h_colors)

    for tip in cfg.skeleton_tips:
        if not tips_coords[tip]:  # catch possible errors
            continue

        color = _get_tip_colormap(tip, len(tips_coords[tip]))
        _tips_ = np.array(tips_coords[tip])

        if cfg.temporal_trails == "lines":
            tips_visuals[tip].set_data(_tips_, color=color, width=cfg.w_visuals)
        elif cfg.temporal_trails == "markers":
            tips_visuals[tip].set_data(_tips_, face_color=color, edge_color=color, size=cfg.w_visuals)

    if not cfg.dataset_create_mode:
        info.text = f"-- {idx_sequence+stt_lim} -- \n{ds_a_labels[idx_sequence]}\n[{canvasVO}] sVO"

    idx_skeleton += 1


def run_threaded_update(n):
    for i in range(n):
        ev = Event("dummy_event")
        update(ev)


def run_app(app):
    if sys.flags.interactive != 1:
        app.run()


def _create_sequence_png(dir):
    global pause
    fname = ds_a_labels[idx_sequence]
    prefix = f"{fname}-"

    if cfg.dataset_create_mode:
        i_subset, i_action, i_id = fname.split(sep="-", maxsplit=2)
        dir = dir.joinpath(f"{i_subset}/{i_action}/{i_id}")
        prefix = ""
    dir.mkdir(exist_ok=True, parents=True)

    # ---
    pause = not pause
    for v in cfg.v_orientation:
        _set_view(v)
        img = canvas.render()

        if v == "side-right":
            img = Image.fromarray(img).rotate(angle=90)
        elif v == "side-left":
            img = Image.fromarray(img).rotate(angle=-90)

        vispy.io.write_png(f"{dir}/{prefix}{v}.png", img)
    pause = not pause
    # ---

    _set_view(canvasVO)
    if not cfg.dataset_create_mode or verbose:
        print(f"> sequence <{idx_sequence+stt_lim}> saved to <{dir}\\{prefix}*.png>")


def get_camera_details():
    print(
        f">>> @i{idx_sequence+stt_lim}-{ds_a_labels[idx_sequence]}:",
        f"elevation {view.camera.elevation}",
        f"| azimuth {view.camera.azimuth}",
        f"| scale {round(view.camera.scale_factor, 2)}",
        # f"| center {np.array(view.camera.center).astype(int)}",
    )


# [INIT VISPY SCENE OBJECTS]___________________________________________________
# build canvas, add viewbox and initialize visuals
canvas = scene.SceneCanvas(
    keys="interactive",
    title="create-imgs-SBUKId",
    app="PyQt5",
    always_on_top=False,
    vsync=False,
    bgcolor="black",
    decorate=False,  # also sets `resizable=False`
    position=[25, 25],
)

view = canvas.central_widget.add_view()
view.camera = "turntable"
view.camera.fov, view.camera.roll, view.camera.distance = 0.0, 0.0, 0.0

skeleton = scene.visuals.Line()
view.add(skeleton)
joint_nodes = scene.visuals.Markers()
view.add(joint_nodes)
head_nodes = scene.visuals.Markers()
view.add(head_nodes)
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
    global idx_sequence, idx_skeleton, pause, tips_coords, save_sequence, canvas, canvasVO

    if event.key == "Escape" and (not cfg.dataset_create_mode):
        print(">>> exiting visualization GUI normally...")
        app.quit()
        return

    elif event.key in ["Left", "Right"] and (not cfg.dataset_create_mode):
        if event.key == "Left":
            canvasVO = cfg.allVOs[(cfg.allVOs.index(canvasVO) - 1) % len(cfg.allVOs)]
        elif event.key == "Right":
            canvasVO = cfg.allVOs[(cfg.allVOs.index(canvasVO) + 1) % len(cfg.allVOs)]

        _set_view(canvasVO)
        info.text = f"-- {idx_sequence+stt_lim} -- \n{ds_a_labels[idx_sequence]}\n[{canvasVO}] sVO"
        return

    elif event.key == " " and (not cfg.dataset_create_mode):
        pause = not pause
        return

    elif event.key == "S" and (not cfg.dataset_create_mode):
        save_sequence = True
        return

    elif event.key == "C" and (not cfg.dataset_create_mode):
        get_camera_details()
        return

    elif event.key in ["P", "N", "Up", "Down"] and (not cfg.dataset_create_mode):
        idx_skeleton = 0

        # skipping to the [P]revious action sequence in the dataset
        if (event.key in ["P", "Down"]) and (idx_sequence > 0):
            idx_sequence -= 1

        # skipping to the [N]ext action sequence in the dataset
        elif (event.key in ["N", "Up"]) and (idx_sequence < len(ds_a_sequences)):
            idx_sequence += 1

        del tips_coords
        tips_coords = {tip: [] for tip in cfg.skeleton_tips}
        print(f"\t -> action {idx_sequence+stt_lim}: {ds_a_labels[idx_sequence]}") if verbose else None

    else:
        return


class loadConfigArguments:
    def __init__(self) -> None:
        config = json.load(open("./modules/.configs/sbukid-v5-default.hgr-config"))
        self.allVOs = ["top-down", "front-to", "front-away", "side-right", "side-left", "custom"]

        self.connection_map = config["connection_map"]
        self.skeleton_tips = config["skeleton_tips"]
        self.skeleton_colors = dict(config["skeleton_colors"])
        self.ds_pckl_file = config["dataset_pickle_file"]

        self.dataset_create_mode = config["dataset_create_mode"]
        self.create_test_subset = config["create_test_subset"]
        self.vo_temporal_gradations = config["add_vo_temporal_gradations"]
        self.vo_skeletons = config["add_vo_skeletons"]
        self.temporal_trails = config["temporal_trails"]
        self.n_joint_coordinates = config["n_joint_coordinates"]

        self.w_visuals = config.setdefault("w_visuals", 3.5)
        self.sz_canvas = config.setdefault("sz_canvas", 960)
        self.s_fitting = config.setdefault("sequence_fitting", "adaptive-mean")
        self.dataset_type = config.setdefault("dataset_type_norm_orig", "norm")
        self.v_orientation = self.allVOs

        if self.dataset_create_mode:
            self.fps = np.inf
            self.n_lims = int(os.cpu_count()) - 2
            self.imgs_ds_dir = Path(
                config["images_dataset_directory"]
                + f"-{self.dataset_type}.{self.sz_canvas}px-[allVOs.adaptiveMean]"
            )
        else:
            self.fps = 1000
            self.n_lims = 1
            self.imgs_ds_dir = Path("./images")

    # ---
    def init_ds_sequences_and_labels(self, data):
        self.ds_a_sequences = np.concatenate(
            (data[f"X_{self.dataset_type}_train"], data[f"X_{self.dataset_type}_valid"]), axis=0
        )
        self.ds_a_labels = np.concatenate(
            (
                "train-" + np.array(data["labels_train"], dtype=object),
                "valid-" + np.array(data["labels_valid"], dtype=object),
            )
        )

        self.ds_a_sequences *= self.sz_canvas  # **tweak here**
        l_dataset, n_skeletons, n_joints, n_coords = self.ds_a_sequences.shape
        assert n_coords == self.n_joint_coordinates, "AssertionError! Invalid ds_a_sequences shape!"
        return l_dataset, n_skeletons, self.ds_a_labels

    # ---
    def get_class_distribution(self):
        classDistribution = {}
        for _lbl in self.ds_a_labels:
            _lbl = _lbl.split("-")[1]
            if _lbl not in classDistribution:
                classDistribution[_lbl] = 1
            else:
                classDistribution[_lbl] += 1

        print(f">>> {classDistribution = } [{len(classDistribution)}]")

    # ---
    def do_sequence_fitting(self):
        a, b, c, d = self.ds_a_sequences.shape
        padding = -(self.sz_canvas / 2.5)  # **tweak here**

        mins = np.min(np.min(np.min(self.ds_a_sequences, axis=3), axis=2), axis=1)
        maxes = np.max(np.max(np.max(self.ds_a_sequences, axis=3), axis=2), axis=1)
        means = np.mean(np.mean(self.ds_a_sequences, axis=2), axis=1).reshape((a, 1, 1, d))

        self.ds_a_sequences -= means - cfg.sz_canvas / 2
        diffs = (maxes - mins + padding).round(2)
        return self.ds_a_sequences, diffs

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
    global ds_a_sequences, ds_a_labels, diffs, stt_lim, canvas
    (stt_lim, end_lim), n = stt_end_args

    ds_a_sequences = ds_a_sequences[stt_lim:end_lim]
    ds_a_labels = ds_a_labels[stt_lim:end_lim]
    diffs = diffs[stt_lim:end_lim]

    if not cfg.dataset_create_mode and verbose:
        print(f"\t@pool {n:02}: {stt_lim=:04} -> {end_lim=:04} == {end_lim-stt_lim} sequences")

    _set_canvas(cfg.sz_canvas)
    _set_view(canvasVO)
    _make_colored_fingers()
    timer = app.Timer(interval=1.0 / cfg.fps, connect=update, start=True)

    if cfg.dataset_create_mode:
        thread = Thread(target=run_app, args=(app,))
        thread.start()
        run_threaded_update((end_lim - stt_lim + 1) * n_skeletons)
        thread.join()
        return None
    else:
        canvas.show(visible=True)
        run_app(app)
        return get_camera_details()


# [>>>>>]____________________________________________________________
# init, init, init
stt_timer = datetime.now()
cfg = loadConfigArguments()

# finger tips history initializations
tips_coords = {tip: [] for tip in cfg.skeleton_tips}

if cfg.temporal_trails == "lines":
    tips_visuals = {tip: scene.visuals.Line() for tip in cfg.skeleton_tips}
else:
    tips_visuals = {tip: scene.visuals.Markers() for tip in cfg.skeleton_tips}

for visual in tips_visuals.values():
    view.add(visual)

# setting joint colors and transparencies
N = len(cfg.connection_map) * 2
s_colors = np.ones((N, 4), dtype=np.float32)
h_colors = np.ones((2, 4), dtype=np.float32)

# init global variables
idx_sequence = idx_skeleton = 0
pause = save_sequence = False
verbose = False
canvasVO = "front-to"

# load `dataset_pickle_file`
pckl_file = open(cfg.ds_pckl_file, "rb")
data = pickle.load(pckl_file, encoding="latin1")
pckl_file.close()

# init action sequences/labels and scene parameters
l_dataset, n_skeletons, ds_a_labels = cfg.init_ds_sequences_and_labels(data)
ds_a_sequences, diffs = cfg.do_sequence_fitting()


# [>>>>>]____________________________________________________________
if __name__ == "__main__":
    print("-" * 50)
    time_elapsed = str(datetime.now() - stt_timer)[2:-7]
    print(f">>> @{time_elapsed}m; dataset and config files loaded successfully...")
    print(f">>> plotting <{ds_a_sequences.shape= }> dataset action sequences...")
    cfg.get_class_distribution()
    print(f">>> viewOrientations = {cfg.v_orientation} [{len(cfg.v_orientation)}]")
    print("-" * 50)

    if cfg.imgs_ds_dir.exists() and cfg.dataset_create_mode:
        print(f"<<< Warning: {cfg.imgs_ds_dir=} exists! Deleting...")
        shutil.rmtree(cfg.imgs_ds_dir, ignore_errors=True)

    stt_timer = datetime.now()
    lims = np.linspace(0, l_dataset, cfg.n_lims + 1, dtype=int)
    starts_ends = [list(lims[i : i + 2]) for i in range(cfg.n_lims)]

    with Pool(cfg.n_lims) as pool:
        pool.map(main, zip(starts_ends, list(range(cfg.n_lims))))

    if cfg.dataset_create_mode and cfg.create_test_subset:
        cfg.do_train_valid_test_split()

    if cfg.dataset_create_mode:
        time_elapsed = str(datetime.now() - stt_timer)[:-7]
        print(f">>> @0{time_elapsed}h; <{len(ds_a_sequences)=}> encoded dataset generated!")
        print(f">>> @{cfg.imgs_ds_dir}")

# [>>>>>]____________________________________________________________
# python modules/create_imgs_v5_SBUKId_mVOs.py
# >>> @00:00:37h; <len(ds_a_sequences)=282> encoded dataset generated!
# >>> @images_d\SBUKId-3D.8G-norm.960px-[allVOs.adaptiveMean]
