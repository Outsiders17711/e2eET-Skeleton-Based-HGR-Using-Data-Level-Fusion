# e2eET Skeleton Based HGR Using Data-Level Fusion
# Dynamic Hand Gestures Classification: Live Stream Demo
# pyright: reportGeneralTypeIssues=false
# pyright: reportWildcardImportFromLibrary=false
# pyright: reportOptionalMemberAccess=false
# -----------------------------------------------

import re
import json
import sys
from pathlib import Path
from datetime import datetime
from vispy import app, scene


# [INIT VISPY SCENE OBJECTS]___________________________________________________
canvas = scene.SceneCanvas(
    keys="interactive",
    title="e2eET HGR: Output GUI",
    app="PyQt5",
    always_on_top=True,
    vsync=False,
    bgcolor="black",
    decorate=True,  # also sets `resizable=False`
    size=(850, 315),
    position=(240, 725),
)

view = canvas.central_widget.add_view()
view.camera = "panzoom"
view.camera.center, view.camera.zoom, view.camera.pan = (0.0, 0.0), 1.0, (0.0, 0.0)

output = scene.Text(
    text="",
    parent=canvas.scene,
    color="darkorange",
    bold=True,
    font_size=9.5,
    anchor_x="left",
    anchor_y="top",
    face="Calibri",
    pos=(25, (canvas.size[1] - 15)),
)


def update(null):
    global output

    with open(cfg.hgr_log) as f:
        output.text = "\n".join(f.readlines())


def run_app(app):
    if sys.flags.interactive != 1:
        app.run()


def _backup_v1():
    if cfg.hgr_log.exists():
        with open(cfg.hgr_log, "r") as f:
            _backup = len(f.readlines()) >= 24  # >= three gs data (8 lines each)

        if _backup:
            tag = re.sub("[-:]", "", str(datetime.now())[:-7]).replace(" ", ".")
            cfg.hgr_log.replace(f"{cfg.hgr_archive}/{cfg.hgr_log.stem}.{tag}.bak")

    open(cfg.hgr_log, "w")


def backup():
    if cfg.hgr_log.exists():
        with open(cfg.hgr_log, "r") as f:
            _contents = f.readlines()
            _backup = len(_contents) >= 24  # >= three gs data (8 lines each)

        if _backup:
            with open(str(cfg.hgr_log).replace("yml", "bak"), "a") as f:
                f.write(f"{'*'*75}\n\n\n")
                f.writelines(_contents)

    with open(cfg.hgr_log, "w") as f:
        f.write(f"--- e2eET HGR Log @ {str(datetime.now())} ".ljust(75, "-"))
        f.write("\n")


@canvas.events.key_press.connect
def on_key_press(event):
    if cfg.debug_mode:
        if event.key == "Escape":
            print("INFO: Exiting visualization GUI normally.")
            app.quit()
            get_camera_details()
            return

        elif event.key == "C":
            get_camera_details()
            return
    return


# [>>>>>]____________________________________________________________
def get_camera_details():
    print(
        f"@get_scene_details():",
        f"\n  {canvas.size=}",
        f"| {canvas.position=}",
        f"\n  {view.camera.center=}",
        f"| {view.camera.zoom=}",
        f"| {view.camera.pan=}",
    )


class loadConfigArguments:
    def __init__(self) -> None:
        config = json.load(open("./allConfigs.jsonc"))
        self.debug_mode = config["debug_mode"]
        self.hgr_archive = Path(config["hgr_archive"])
        self.hgr_log = Path(config["hgr_log"])

        self.fps = 5
        self.hgr_archive.mkdir(exist_ok=True)


def main():
    backup()
    timer = app.Timer(interval=(1.0 / cfg.fps), connect=update, start=True)
    print("INFO: Initialized <vispyOutputGUI.py> ...")

    canvas.show(visible=True)
    run_app(app)


# [>>>>>]____________________________________________________________
if __name__ == "__main__":
    cfg = loadConfigArguments()
    main()
