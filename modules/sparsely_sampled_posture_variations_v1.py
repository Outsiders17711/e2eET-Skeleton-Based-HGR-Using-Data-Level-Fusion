# %%
import pickle
from pathlib import Path

import numpy as np
import cv2 as cv


class sparselySampledSkeletons:
    def __init__(self, sz_canvas, connection_map, use_flow=False, n_samples=9, scale_factor=1000):
        self.sz_canvas = sz_canvas
        self.connection_map = connection_map
        self.scale_factor = scale_factor
        self.use_flow = use_flow

        assert n_samples in [4, 9, 16, 25], "valid `n_samples` options are [4, 9, 16, 25]"
        self.n_samples = n_samples
        self.grads = np.linspace(255 / n_samples, 255, n_samples, dtype=int).tolist()

        self.a_border = 5  # hardcoded!
        self.m_scale = 5  # hardcoded!

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

    def createSpSaSkels(self, sequence, savepath, prefix=None, debug=False):
        savepath.mkdir(exist_ok=True, parents=True)
        prefix = prefix + "sss" if prefix else "sp_sa_sk"

        imgs = []
        samples = np.linspace(0, len(sequence) - 1, self.n_samples, dtype=int)

        # [NOTE]: `_prev_img` and `img` must be grayscale images
        for idx, n in enumerate(samples):
            _img = self._one_skeleton(sequence[n], self.grads[idx])
            if self.use_flow:
                _prev_img = self._one_skeleton(sequence[n - 1], self.grads[idx]) if n else None
                _img = self._draw_flow(_img, _prev_img, self.grads[idx])

            cv.imwrite(f"{savepath}/{prefix}_{idx:02}.png", _img) if debug else None
            imgs.append(_img)

        n_stk = int(np.sqrt(self.n_samples))
        stack = np.vstack([np.hstack(imgs[i * n_stk : (i * n_stk) + n_stk]) for i in range(n_stk)])
        stack = cv.resize(stack, None, fx=(1 / n_stk), fy=(1 / n_stk))  # type:ignore
        # stack = cv.cvtColor(stack, cv.COLOR_GRAY2RGB)  # fastai loads images as "RGB" by default

        cv.imwrite(f"{savepath}/{prefix}_stk0.png", stack)
        cv.imshow("sparselySampledSkeletons", stack) if debug else None

        _, no_grad_stack = cv.threshold(stack, 0, 255, cv.THRESH_BINARY)
        cv.imwrite(f"{savepath}/{prefix}_stk1.png", no_grad_stack)
        cv.imshow("noGraduations", no_grad_stack) if debug else None


# %%
# [TEST]_______________________________________________________________________
if __name__ == "__main__":
    img_temp_dir = Path("./images")
    pckl_file = open("./datasets/dhg1428_3d_dictTTS_l250.pckl", "rb")
    data = pickle.load(pckl_file, encoding="latin1")
    pckl_file.close()

    sequences = np.abs(data["X_valid"])
    sequences = sequences.reshape((*sequences.shape[:2], 22, -1))
    s = sequences[0]
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

    sss = sparselySampledSkeletons(640, connection_map, scale_factor=s_scale, n_samples=9)
    cv.imshow("oneSkeleton", sss._one_skeleton(s[0], 255))
    sss.createSpSaSkels(s, img_temp_dir, debug=True)

    _s = (s[0] * s_scale).astype(int)[:, :2]
    img = np.zeros((640, 640), dtype=np.uint8)
    for (pt1, pt2) in connection_map:
        cv.line(img, _s[pt1], _s[pt2], (255, 0, 255), 1)
    for pts in _s:
        cv.circle(img, pts, 2, (255, 255, 255), cv.FILLED)
    cv.imshow("rawSkeleton", img)

    cv.waitKey(0)
    cv.destroyAllWindows()
