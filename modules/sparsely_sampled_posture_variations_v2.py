import pickle
from pathlib import Path

import numpy as np
import cv2 as cv


# [>>>>>]____________________________________________________________
class sparselysampledPostureVariations:
    def __init__(
        self,
        sz_canvas,
        connection_map,
        grid_sizes,
        scale_factor,
        dataset_create_mode,
        add_sspv_temporal_gradations=True,
        add_optical_flow=False,
        save_grid_components=False,
    ):
        # ---
        self.sz_canvas = sz_canvas
        self.connection_map = connection_map
        self.grid_sizes = grid_sizes
        self.scale_factor = scale_factor
        self.dataset_create_mode = dataset_create_mode
        self.add_optical_flow = add_optical_flow
        self.save_grid_components = save_grid_components
        # ---
        self.a_border = 5  # hardcoded!
        self.m_scale = 5  # hardcoded!
        self.n_samples = max(grid_sizes)
        # ---
        if add_sspv_temporal_gradations:
            self.grads = np.linspace(255 / self.n_samples, 255, self.n_samples, dtype=int).tolist()
        else:
            self.grads = [255] * self.n_samples

    def _one_skeleton(self, skeleton, grad_value):
        # basic checks; catch errors
        assert len(skeleton.shape) == 2, "invalid skeleton shape!"
        skeleton = skeleton[:, :2]  # ensure 2d data

        # create a fit between the skeleton and background image
        skeleton = (skeleton * self.scale_factor).astype(int)
        mins = np.min(skeleton, axis=0)
        skeleton = (skeleton - mins + self.a_border) * self.m_scale
        maxes = np.max(skeleton, axis=0)
        lims = np.flip((maxes + (self.m_scale * self.a_border)))

        # draw the skeleton on background image
        img = np.zeros(lims, dtype=np.uint8)
        for (pt1, pt2) in self.connection_map:
            cv.line(img, skeleton[pt1], skeleton[pt2], grad_value, self.m_scale)
        for pts in skeleton:
            cv.circle(img, pts, int(self.m_scale * 1.5), grad_value, cv.FILLED)

        # flip, resize and return drawn skeleton image
        return cv.flip(cv.resize(img, (self.sz_canvas, self.sz_canvas)), 1)

    def _draw_flow(self, img, prev_img, grad_value, sz_grid=15, min_abs_flow=0.01):
        # calculate optical flow
        prev_img = img.copy() if prev_img is None else prev_img
        flow = cv.calcOpticalFlowFarneback(prev_img, img, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # create flow field (grid)
        h, w = img.shape[:2]
        yx = np.mgrid[sz_grid / 2 : h : sz_grid, sz_grid / 2 : w : sz_grid].reshape(2, -1)
        y, x = yx.astype(int)
        fx, fy = flow[y, x].T

        # filter out flow vectors with minimal movement
        for idx, (x_val, y_val) in enumerate(zip(fx, fy)):
            if abs(x_val) < min_abs_flow or abs(y_val) < min_abs_flow:
                x[idx] = y[idx] = 0

        # draw the flow vectors
        lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
        lines = (lines - 1.5).astype(int)  # shift the grid around the image
        cv.polylines(img, lines, 0, grad_value, thickness=1)

        return img

    def one_sspv_sequence(self, sequence, savepath, disp_title):
        imgs = []
        samples = np.linspace(0, len(sequence) - 1, self.n_samples, dtype=int)

        # [NOTE]: `_prev_img` and `img` must be grayscale images
        for idx, n in enumerate(samples):
            _img = self._one_skeleton(sequence[n], self.grads[idx])

            if self.add_optical_flow:
                _prev_img = self._one_skeleton(sequence[n - 1], self.grads[idx]) if n else None
                _img = self._draw_flow(_img, _prev_img, self.grads[idx])

            imgs.append(_img)
            if self.dataset_create_mode and self.save_grid_components:
                cv.imwrite(f"{savepath}/ssPV-{idx:02}.png", _img)

        # ---
        imgs = np.array(imgs)
        for _dim in self.grid_sizes:
            if _dim == 1:
                print(">>> ValueWarning: grid size of 1x1 is invalid, skipping...")
                continue

            _idxs = np.linspace(0, self.n_samples - 1, _dim ** 2, dtype=int)
            grid = imgs[_idxs]
            grid = np.vstack([np.hstack(grid[i * _dim : (i * _dim) + _dim]) for i in range(_dim)])
            grid = cv.resize(grid, None, fx=(1 / _dim), fy=(1 / _dim))  # type:ignore

            if self.dataset_create_mode:
                cv.imwrite(f"{savepath}/ssPV-grid-{_dim}x{_dim}.png", grid)
            else:
                cv.imshow(f"{disp_title}-grid-{_dim}x{_dim}", grid)
                cv.waitKey(1)


# [TEST]_______________________________________________________________________
if __name__ == "__main__":
    # --- load dataset
    img_temp_dir = Path("./images")
    pckl_file = open("./datasets/dhg1428_3d_dictTTS_l250.pckl", "rb")
    data = pickle.load(pckl_file, encoding="latin1")
    pckl_file.close()

    # --- set ssPV parameters
    idx = 0
    sequences = np.abs(data["X_valid"])
    sequences = sequences.reshape((*sequences.shape[:2], 22, -1))
    s = sequences[idx]
    s_scale = 1000 if sequences.shape[-1] == 3 else 1

    connection_map = [
        [0, 2],
        [2, 3],
        [3, 4],
        [4, 5],
        [0, 1],
        [1, 6],
        [6, 7],
        [7, 8],
        [8, 9],
        [1, 10],
        [10, 11],
        [11, 12],
        [12, 13],
        [1, 14],
        [14, 15],
        [15, 16],
        [16, 17],
        [1, 18],
        [18, 19],
        [19, 20],
        [20, 21],
    ]

    sspvs = sparselysampledPostureVariations(
        sz_canvas=640,
        connection_map=connection_map,
        grid_sizes=[2, 3, 4, 5],
        scale_factor=s_scale,
        dataset_create_mode=False,
        add_sspv_temporal_gradations=True,
        add_optical_flow=False,
        save_grid_components=False,
    )

    # --- create ssPVs
    cv.imshow("oneSkeleton", sspvs._one_skeleton(s[0], 255))
    sspvs.one_sspv_sequence(s, img_temp_dir, disp_title=data["valid_str_labels"][idx])

    # --- display raw gesture sequence for comparison
    _s = (s[idx] * s_scale).astype(int)[:, :2]
    img = np.zeros((640, 640), dtype=np.uint8)
    for (pt1, pt2) in connection_map:
        cv.line(img, _s[pt1], _s[pt2], (255, 0, 255), 1)
    for pts in _s:
        cv.circle(img, pts, 2, (255, 255, 255), cv.FILLED)
    cv.imshow("rawSkeleton", img)

    cv.waitKey(0)
    cv.destroyAllWindows()
