# pyright: reportGeneralTypeIssues=false
# pyright: reportOptionalMemberAccess=false
# pyright: reportWildcardImportFromLibrary=false
# -----------------------------------------------
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
if os.name == "nt": os.EX_OK = 0
# ---
import random
import shutil
import traceback
from textwrap import dedent
from datetime import datetime
# ---
import torch
import torch.nn as NN
import numpy as np
from fastai.vision.all import *
from fastai.torch_core import defaults
from torch.utils.tensorboard.writer import SummaryWriter
from IPython.display import clear_output
from sklearn.model_selection import train_test_split
# ---
from __main__ import args, deets
from exp00_helper_functions import *


__all__ = [
    "multiOrientationDataLoader",
    "multiOrientationModel",
    "modelCheckpoint",

    "initTraining",
    "iterativeScaling",
    "i_Timer",
    "Logger",
    "i_Logger",
    "Cleaner",
    "generateModelGraph",

    "opt_InitItrFinetune",
    "itr_scl_sizes",
    "e_stt_datetime",
    "e_strftime",

    "ensembleTunerDataloader",
    "ensembleTunerTraining",
    "ensembleTunerEvaluation",
    "tunerImages",

    "end2endTunerModel",
    "e2eTunerLossWrapper",
    "attachMetrics",
    "returnMetrics",
    "e2eTunerLearnerEvaluation",

    "tensorTunerDataloader",
    "tensorTunerModel_CNN",
    "tensorTunerModel_FCN",
    "tensorTunerTraining",
    "tensorTunerEvaluation",
]
# -----------------------------------------------


e_repr_seed = 17711
e_stt_datetime = datetime.now()
e_strftime = e_stt_datetime.strftime("%d%m%y.%H%M")
if not(hasattr(args, "e_history")): setattr(args, "e_history", [])
e_args = "\n".join([f"\t{_arg.ljust(25)} -- {str(type(getattr(args, _arg))).ljust(15)} -- {getattr(args, _arg)}" for _arg in vars(args)])

e_details = f"{deets.e_tag=}\n{deets.e_secret=}\n{e_strftime=}\n{deets.e_desc=}\n{deets.ds_directory=}\ne_args=\n{e_args}"
deets = deets._replace(e_repr_seed=e_repr_seed, e_stt_datetime=e_stt_datetime, e_strftime=e_strftime, e_details=e_details)
if args.verbose: print(e_details, end=f"\n{'-'*15}\n") ; os._exit(os.EX_OK)
# -----------------------------------------------


opt_pctStart = {
    "2d": {3: 0.80, 5: 0.80, 10: 0.75, 15: 0.20, 20: 0.25, 25: 0.75, 30: 0.05, 35: 0.55, 40: 0.60},
    "3d": {3: 0.75, 5: 0.80, 10: 0.20, 15: 0.30, 20: 0.55, 25: 0.35, 30: 0.60, 35: 0.05, 40: 0.45},
}

opt_InitEpochs = {"2d": 20, "3d": 20}
opt_InitItrFinetune = {"2d": True, "3d": False}
opt_ItrScalingSizes = {"2d": [224, 224, 276, 328], "3d": [276, 328, 380]}
itr_scl_sizes = opt_ItrScalingSizes[args.nd] if args.itr_scl_sizes == ["<OPT>"] else args.itr_scl_sizes

vocab_DHG1428 = {
    14: ["01_Grab", "02_Tap", "03_Expand", "04_Pinch", "05_RotationCW", 
        "06_RotationCCW", "07_SwipeRight", "08_SwipeLeft", "09_SwipeUp",
        "10_SwipeDown", "11_SwipeX", "12_Swipe+", "13_SwipeV", "14_Shake"],
    28: ["01_Grab", "02_Tap", "03_Expand", "04_Pinch", "05_RotationCW", "06_RotationCCW", 
        "07_SwipeRight", "08_SwipeLeft", "09_SwipeUp", "10_SwipeDown", "11_SwipeX", 
        "12_Swipe+", "13_SwipeV", "14_Shake", "15_Grab", "16_Tap", "17_Expand", 
        "18_Pinch", "19_RotationCW", "20_RotationCCW", "21_SwipeRight", "22_SwipeLeft", 
        "23_SwipeUp", "24_SwipeDown", "25_SwipeX", "26_Swipe+", "27_SwipeV", "28_Shake"]
}
# -----------------------------------------------


def set_reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    #
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None
    torch.cuda.manual_seed(seed) if torch.cuda.is_available() else None
    #
    g = torch.Generator()
    g.manual_seed(seed)
    #
    def e_seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    return g, e_seed_worker

e_repr_gen, e_seed_worker = set_reproducibility(seed=deets.e_repr_seed)
no_random(seed=deets.e_repr_seed, reproducible=True)
# -----------------------------------------------


class ImageTuples(fastuple):
    @classmethod
    def create(cls, fns): return cls(tuple(PILImage.create(f) for f in fns))

    def show(self, ctx=None, **kwargs):
        imgs = list(self)
        for i in imgs:
            if (not isinstance(i, Tensor)) or (i.shape != imgs[0].shape):
                imgs = [i.resize(imgs[0].size) for i in imgs]
                imgs = [tensor(i).permute(2, 0, 1) for i in imgs]
                break

        line = imgs[0].new_zeros(imgs[0].shape[0], imgs[0].shape[1], 5)
        line[:] = 255

        for idx in range(len(imgs)):
            ins_idx = (idx * 2) + 1
            imgs.insert(ins_idx, line) if ins_idx < len(imgs) else None

        return show_image(torch.cat(imgs, dim=2), figsize=[2.5 * len(imgs)] * 2, ctx=ctx, **kwargs)

def ImageTupleBlock():
    return TransformBlock(type_tfms=ImageTuples.create, batch_tfms=IntToFloatTensor)

class e2eTunerImageTuples(fastuple):
    @classmethod
    def create(cls, fns): 
        imgs = tuple(PILImage.create(f) for f in fns)
        # [ENCODE] convert label string to a list(int) of ASCII values to bypass fastai checks
        label = tuple([ord(c) for c in format(fns[0].parent.parent.name, "32")])
        return cls(tuple((imgs, label)))

    def show(self, ctx=None, **kwargs):
        # [DECODE] unzip tuple and translate list(int) of ASCII values to label string (not used)
        imgs, label = list(self[0]), "".join(map(chr, self[1])).replace(" ", "")
        # print(f"{label= }")
        
        for i in imgs:
            if (not isinstance(i, Tensor)) or (i.shape != imgs[0].shape):
                imgs = [i.resize(imgs[0].size) for i in imgs]
                imgs = [tensor(i).permute(2, 0, 1) for i in imgs]
                break

        line = imgs[0].new_zeros(imgs[0].shape[0], imgs[0].shape[1], 5)
        line[:] = 255

        for idx in range(len(imgs)):
            ins_idx = (idx * 2) + 1
            imgs.insert(ins_idx, line) if ins_idx < len(imgs) else None

        return show_image(torch.cat(imgs, dim=2), figsize=[2.5 * len(imgs)] * 2, ctx=ctx, **kwargs)

def e2eTunerImageTupleBlock():
    return TransformBlock(type_tfms=e2eTunerImageTuples.create, batch_tfms=IntToFloatTensor)

def get_gesture_sequences(path):
    files = get_image_files(path)
    return L(dict.fromkeys([f.parent for f in files]))

def get_orientation_images(o):
    return [(o / f"{_vo}.png") for _vo in args.mv_orientations]

def get_mVOs_img_size(subset):
    assert "fastai.data.core.TfmdDL" in str(type(subset)), "train/valid dls subset should be provided!"
    return PILImage.create(f"{subset.items[0]}/{args.mv_orientations[0]}.png").size

@typedispatch
def show_batch(x:e2eTunerImageTuples, y, samples, ctxs=None, max_n=12, nrows=3, ncols=2, figsize=None, **kwargs):
    if figsize is None: figsize = (ncols*8, nrows*3)
    if ctxs is None: ctxs = get_grid(min(len(samples), max_n), nrows=nrows, ncols=ncols, figsize=figsize)
    # ---
    ctxs = show_batch[object](x, y, samples, ctxs=ctxs, max_n=max_n, **kwargs)  # type:ignore
    return ctxs

@typedispatch
def show_batch(x:ImageTuples, y, samples, ctxs=None, max_n=12, nrows=3, ncols=2, figsize=None, **kwargs):
    if figsize is None: figsize = (ncols*8, nrows*3)
    if ctxs is None: ctxs = get_grid(min(len(samples), max_n), nrows=nrows, ncols=ncols, figsize=figsize)
    # ---
    ctxs = show_batch[object](x, y, samples, ctxs=ctxs, max_n=max_n, **kwargs)  # type:ignore
    return ctxs

def multiOrientationDataLoader(ds_directory, bs, img_size, shuffle=True, return_dls=True, ds_valid="valid", e2eTunerMode=False, preview=False, _e_seed_worker=None, _e_repr_gen=None):
    tfms = aug_transforms(
        do_flip=True, flip_vert=False, max_rotate=25.0, max_zoom=1.5, 
        max_lighting=0.5, max_warp=0.1, p_affine=0.75, p_lighting=0.75,
    )

    multiDHG1428 = DataBlock(
        blocks=((e2eTunerImageTupleBlock if e2eTunerMode else ImageTupleBlock), CategoryBlock),
        get_items=get_gesture_sequences,
        get_x=get_orientation_images,
        get_y=parent_label,
        splitter=GrandparentSplitter(train_name="train", valid_name=ds_valid),
        item_tfms=Resize(size=img_size, method=ResizeMethod.Squish),
        batch_tfms=[*tfms, Normalize.from_stats(*imagenet_stats)],
    )

    ds = multiDHG1428.datasets(ds_directory, verbose=False)
    if return_dls:
        dls = multiDHG1428.dataloaders(ds_directory, bs=bs, worker_init_fn=e_seed_worker, generator=e_repr_gen, device=defaults.device, shuffle=shuffle)
        # clear_output(wait=False)
        assert dls.c == args.n_classes, ">> ValueError: dls.c != n_classes as specified!!"

        if preview:
            print(dedent(f"""
            Dataloader has been created successfully...
            The dataloader has {len(dls.vocab)} ({dls.c}) classes: {dls.vocab}
            Training set [len={len(dls.train.items)}, img_sz={get_mVOs_img_size(dls.train)}] loaded on device: {dls.train.device}
            Validation set [len={len(dls.valid.items)}, img_sz={get_mVOs_img_size(dls.valid)}] loaded on device: {dls.valid.device}
            Previewing loaded data [1] and applied transforms [2]...
            """))
            dls.show_batch(nrows=1, ncols=4, unique=False, figsize=(12, 12))
            dls.show_batch(nrows=1, ncols=4, unique=True, figsize=(12, 12))
        else: clear_output(wait=False)

        return dls

    else: return ds
# -----------------------------------------------


def Logger(*args, stdout=True):
    _e_prefix = f"[{deets.e_secret}]-{deets.e_strftime} ->- {deets.e_desc} ->-"
    with open(f"{deets.e_tag}.yml", "a") as f: print(_e_prefix, *args, file=f)  # write to logfile
    if stdout: print(*args)  # write to std.out, maybe?

def i_Logger(i, e_accuracy):
    arr_metric_values = np.array(args.e_history)[::-1]
    e_accuracy = np.round(e_accuracy, decimals=4)
    idx_e_accuracy = arr_metric_values[:,-1].round(4).tolist().index(e_accuracy)

    str_metric_values = ""
    for _mv  in arr_metric_values[idx_e_accuracy][2:]: str_metric_values += f"{_mv:.4f} - "
    
    Logger(f"-- [{i:>6}] - {str_metric_values}0{str(datetime.now()-deets.e_stt_datetime)[:-10]}")

def i_Timer(stt_time, stdout=True):
    diff = str(datetime.now() - stt_time).split(":")
    diff = f"0{diff[0]}h:{diff[1]}m:{diff[2][:2]}s"
    
    if stdout: print(f"Time Elapsed ~ {diff}")
    else:      return diff

def Cleaner(target=None):
    def _cleanup(_f):
        try:    os.rmdir(_f)
        except: shutil.rmtree(_f, ignore_errors=True)

    if target: _cleanup(target)

    for _f in Path(f"../models").iterdir():
        if _f.is_dir() and not(list(_f.iterdir())) and not(_f.name.endswith("checkpoint")):
            _cleanup(_f)
# -----------------------------------------------


class outsidersCustomCallback(TrackerCallback):
    order = TrackerCallback.order + 4

    def __init__(self, i_tag, e_epochs, e_accuracy, monitor, verbose=False):
        super().__init__(monitor=monitor)
        store_attr()

        self.log_dir = f"../runs/{deets.e_tag}/[{deets.e_secret}]-{deets.e_strftime}-{deets.e_desc}"
        self._tmp_model_tag = f"tmp-{deets.e_desc}-{i_tag}"
        self.i_accuracy = 0.0

    def before_fit(self):
        str_model_path = f"e_model_path: {self.path}/{deets.e_model_tag}"
        if self.verbose: print(str_model_path)

        if args.create_e_tb_events:
            self.writer = SummaryWriter(log_dir=self.log_dir)
            self.writer.add_text("experiment_details", deets.e_details, 0)
            self.writer.add_text("experiment_details", str_model_path, 1)

        super().before_fit()

    def after_epoch(self):      
        super().after_epoch()
        accuracy = self.recorder.values[-1][self.idx]

        if accuracy > self.i_accuracy:
            self.i_accuracy = accuracy
            self.save(self._tmp_model_tag, with_opt=True)

            if self.i_accuracy > self.e_accuracy:
                self.e_accuracy = self.i_accuracy
                self.save(deets.e_model_tag, with_opt=True)  # ensure that best model is saved

            if self.verbose:
                print(f"@{self.i_tag}: Model {self.monitor} improved >> i/{self.epoch:02d}/{self.e_epochs:02d}/{self.i_accuracy:.4f}")

        if args.create_e_tb_events: self.record_metric_values()
        self.e_epochs += 1

    def record_metric_values(self):
        self.writer.add_scalar("expAccuracy", self.e_accuracy, self.e_epochs)
        self.writer.add_scalar("lr", self.opt.hypers[-1]["lr"], self.e_epochs)

        targ_metrics = self.recorder.metric_names[1:-1]
        for _idx, _tm in enumerate(targ_metrics):
            self.writer.add_scalar(f"{_tm}", self.recorder.values[-1][_idx], self.e_epochs)
        
    def after_fit(self, **kwargs):
        self.load(self._tmp_model_tag, with_opt=True)
        if args.create_e_tb_events: self.writer.close()
        if self.verbose: print(f"@cbs {self= }")
        print(f"@outsidersCustomCallback: Training completed and best {self.i_tag} model loaded!")
# -----------------------------------------------


def i_LRFinder(learn, show_plot=False, n_attempts=0):
    try: return learn.lr_find(suggest_funcs=(valley, slide), show_plot=show_plot)

    except Exception as e:
        n_attempts += 1
        print(f"@{n_attempts=}: i_LRFinder Exception:: {e}!")
        if "CUDA" in str(e):
            Logger(f"CUDA RuntimeError - {e}!") ; os._exit(os.EX_OK)
        else:  # "no elements" in str(e) or "out of bounds" in str(e)  or "numerical gradient" in str(e)
            if n_attempts == 69:
                print(f"i_LRFinder Exception::", traceback.format_exc())
                Logger(f"i_LRFinder Exception:: {e}!") ; os._exit(os.EX_OK)
            return i_LRFinder(learn, show_plot, n_attempts)

def i_LRHistorical(n_classes, i_tag):
    maxLearningRates = {
        #  --- 14 [full] ---
        14: {"i224a": 0.001445, "i224b": 0.000063, "276a": 0.001738, "276b": 0.000091, 
            "328a": 0.001445, "328b": 0.000052, "380a": 0.001738, "380b": 0.000063,},
        #  --- 28 [full] ---
        28: {"i224a": 0.002089, "i224b": 0.000091, "276a": 0.002512, "276b": 0.000076, 
            "328a": 0.001445, "328b": 0.000063, "380a": 0.001738, "380b": 0.000063,},
            
        #  --- 13 [14.full.copy] ---
        13: {"i224a": 0.001445, "i224b": 0.000063, "276a": 0.001738, "276b": 0.000091, 
            "328a": 0.001445, "328b": 0.000052, "380a": 0.001738, "380b": 0.000063,},
        #  --- 45 [14.full.copy] ---
        45: {"i224a": 0.001445, "i224b": 0.000063, "276a": 0.001738, "276b": 0.000091, 
            "328a": 0.001445, "328b": 0.000052, "380a": 0.001738, "380b": 0.000063,},
    }

    return maxLearningRates[n_classes][i_tag]

def getLR(learn, i_tag, show_plot=False):
    if args.lrs_type == "lrHistorical": return i_LRHistorical(args.n_classes, i_tag)
    elif args.lrs_type == "lrFinder":   return i_LRFinder(learn, show_plot=show_plot).valley
    else:                               return defaults.lr

def isFrozen(learn):
    for child in learn.model.children():
        for param in child.parameters():
            if param.requires_grad == False:
                print("\nModel FROZEN [a].") ; return "a"

    print("\nModel UNFROZEN [b].") ; return "b"

def FitFlatCosine(learn, i_tag, i_eps, i_pct_start, e_epochs_lr_accuracy, finetune=False, log=False, verbose=False):
    e_epochs, e_lr, e_accuracy = e_epochs_lr_accuracy

    if finetune: i_lr, i_tag = (e_lr / 10), f"{i_tag}f"
    else:        e_lr = i_lr = getLR(learn, i_tag=i_tag)
    print(f">> @{i_tag} --- pct_start: {i_pct_start} --- learning_rate: {i_lr}")

    monitor = "accuracyTuner" if hasattr(learn.model, "tuner_img_sz") else "accuracy"
    oCCb = outsidersCustomCallback(i_tag, e_epochs, e_accuracy, monitor, verbose=verbose)
    learn.fit_flat_cos(n_epoch=i_eps, lr=i_lr, pct_start=i_pct_start, cbs=[oCCb])

    args.e_history.extend(learn.recorder.values)
    if verbose: learn.show_results(nrows=1, ncols=4)
    if log: i_Logger(i_tag, oCCb.i_accuracy)

    return (learn, oCCb.e_epochs, e_lr, oCCb.e_accuracy)
# -----------------------------------------------

def FitFineTune(learn, i_tag, i_eps, i_pct_start, e_epochs_lr_accuracy, log=False, verbose=False):
    (e_epochs, e_lr, e_accuracy), i_lr = e_epochs_lr_accuracy, 0.002
    i_frz_eps, i_ufrz_eps = i_eps
    print(f">> @{i_tag} --- pct_start: {i_pct_start} --- learning_rate: {i_lr}")

    monitor = "accuracyTuner" if hasattr(learn.model, "tuner_img_sz") else "accuracy"
    oCCb = outsidersCustomCallback(i_tag, e_epochs, e_accuracy, monitor, verbose=verbose)
    learn.fine_tune(freeze_epochs=i_frz_eps, epochs=i_ufrz_eps, base_lr=i_lr, pct_start=i_pct_start, cbs=[oCCb])

    args.e_history.extend(learn.recorder.values)
    if verbose: learn.show_results(nrows=1, ncols=4)
    if log: i_Logger(i_tag, oCCb.i_accuracy)

    return (learn, oCCb.e_epochs, e_lr, oCCb.e_accuracy)
# -----------------------------------------------


class multiOrientationModel(Module):
    def __init__(self, encoder, head, nf, debug=False):
        self.encoder = encoder.to(defaults.device)
        self.head = head.to(defaults.device)
        self.debug = debug

        if args.ftr_fusion == "conv":
            _in_channels = nf * len(args.mv_orientations)
            fusion = NN.Sequential(
                NN.Conv2d(_in_channels, nf, kernel_size=(1, 1), stride=(1, 1), bias=False),
                NN.BatchNorm2d(nf, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                NN.ReLU(inplace=True),
            )
            self.head = NN.Sequential(fusion, self.head).to(defaults.device)

        if self.debug:
            print(f"{self.encoder= }", end=f"\n{'-'*50}\n")
            print(f"{self.head= }", end=f"\n{'-'*50}\n")

    def forward(self, X, y=None):
        if y is not None: X = [X, y]  # [NOTE] modified for tensorboard SummaryWriter

        if self.debug:
            print(f"{type(X)= } | {len(X)= } | {explode_types(X)= }")
            for xi in X: print(f"{explode_types(xi)= } | {xi.shape= }")

        X = [self.encoder(xi) for xi in X]
        ftrs = self.ftr_lvl_fusion(X)

        if self.debug:
            for xi in X: print(f">> {xi.shape= }")
            print(f">> {ftrs.shape= }")
            self.debug = False ; os._exit(os.EX_OK)

        return self.head(ftrs)

    def ftr_lvl_fusion(self, X: list):
        if args.ftr_fusion == "sum": return torch.add(*X)
        elif args.ftr_fusion == "max": return torch.max(*X)
        elif args.ftr_fusion == "avg": return torch.add(*X) / 2
        elif args.ftr_fusion in ["cat", "conv"]: return torch.cat(X, dim=1)
        else: assert False, ">> ValueError: unrecognized value passed for feature level fusion type!!"

    def splitter(self, model: Module):
        return [params(model.encoder), params(model.head)]
# -----------------------------------------------


def _split_Xy(b):
    i = 1 if (len(b) == 1) else len(b) - 1
    return b[:i], b[i:]

def _get_predictions(learn, dls, valid_only):
    preds, targs = [], []
    dls_subsets = [1] if valid_only else [0, 1]
    
    try:    _model = learn.model
    except: _model = learn
    
    for idx in dls_subsets:
        for b in dls[idx]: 
            xb, yb = _split_Xy(b)
            with torch.no_grad(): 
                _preds, _targs = _model(*xb), yb
            preds.extend(_preds)
            targs.extend(_targs)

    return torch.vstack(preds), torch.cat(targs, dim=0)

def saveEnsembleTunerPredsTargs(mVOs, ds_directory, pkl_pt_mask, _img_sz=380, _bs=16, verbose=False, e_seed_worker=None, e_repr_gen=None):
    
    for dls_subset in  ["train-valid", "valid", "test"]:
        _valid_only = False if (dls_subset == "train-valid") else True
        _ds_valid = dls_subset if (dls_subset == "test") else "valid"
        _pt_file = f"{pkl_pt_mask}-[{dls_subset}].pt"

        dls = multiOrientationDataLoader(
            ds_directory, bs=_bs, img_size=_img_sz, shuffle=False, ds_valid=_ds_valid,
            # _e_seed_worker=e_seed_worker, _e_repr_gen=e_repr_gen, 
        )
        learn = torch.load(f=f"{pkl_pt_mask}.pkl")

        preds, targs = _get_predictions(learn, dls, valid_only=_valid_only)
        decoded = torch.argmax(preds, dim=1)
        accuracy = (sum(decoded == targs) / len(targs)).item()

        preds = torch.unsqueeze(preds, dim=1)
        X = preds.cpu().numpy()
        y = targs.cpu().numpy()
        torch.save(obj=[X, y], f=_pt_file)

        if verbose:
            print(f"\n>> Processed {translateMVOs(mVOs)} {dls.c}G {dls_subset=}!")
            print(f">> --- {preds.shape= } | {decoded.shape} | {targs.shape} | {accuracy:.4f}\n")

def modelCheckpoint(learn, learn_directory, ds_tuner=True):
    _stem = f"[{deets.e_secret}]-{args.n_classes}G-{translateMVOs(args.mv_orientations)}"
    _path_mask = f"../checkpoints/{_stem}"
    
    learn = learn.load(file=deets.e_model_tag, with_opt=True)
    learn.export(fname=f"{_path_mask}.pkl")

    if ds_tuner:
        saveEnsembleTunerPredsTargs(
            mVOs=args.mv_orientations, ds_directory=deets.ds_directory, 
            pkl_pt_mask=f"{learn_directory}/{_path_mask}", 
            # e_seed_worker=e_seed_worker, e_repr_gen=e_repr_gen, 
            _img_sz=380, verbose=False
        )

    print(f"Learner pkl and pt checkpoints created successfully")
# -----------------------------------------------


class tunerImages(fastuple):
    @classmethod
    def create(cls, Xy): 
        assert len(Xy) == 2, ">> InputError: expected list/tuple if X and y data!!"
        X, y = Xy
        img_size = 560
        l_band = img_size // X.shape[-1]

        imgs = [np.zeros([img_size, img_size, 3], dtype=np.uint8) for vo in X]
        stt_bands = [(np.argmax(vo)*l_band) for vo in X]

        for _idx, (_img, _stt_band) in enumerate(zip(imgs, stt_bands)):
            correct = _stt_band ==  y*l_band
            _img[_stt_band:_stt_band+l_band, :] = [0, 255, 0] if correct else [255, 0, 0]
            imgs[_idx] = _img

        return PILImage.create(np.hstack(imgs))
    
    def show(self, ctx=None, **kwargs): 
        img = self[0]
        if not isinstance(img, Tensor): img = tensor(img)
        return show_image(img, ctx=ctx, figsize=(10, 10), **kwargs)

def tunerImageBlock():
    return TransformBlock(type_tfms=tunerImages.create, batch_tfms=IntToFloatTensor)

def get_tunerPredsTargs(ds_subset):
    global deets
    ds = loadEnsembleTunerDataset(mVOs=args.mv_orientations, verbose=(ds_subset=="test"))

    if ds_subset == "train-valid":
        valid_pct = 0.25
        X_trn_val, y_trn_val = ds["train-valid"]
        rand_idxs = torch.randperm(len(y_trn_val)).numpy()
        X_trn_val, y_trn_val = X_trn_val[rand_idxs], y_trn_val[rand_idxs]

        l_ds_Valid = int(valid_pct * len(y_trn_val))
        l_ds_Train = len(y_trn_val) - l_ds_Valid
        print(f"{X_trn_val.shape=} | {y_trn_val.shape=}")
    
    elif ds_subset == "test":
        X_trn_val, y_trn_val = ds["test"]

        l_ds_Train = l_ds_Valid = len(y_trn_val)
        print(f"{X_trn_val.shape=} | {y_trn_val.shape=}")

        X_trn_val = np.concatenate((X_trn_val, X_trn_val), axis=0)
        y_trn_val = np.concatenate((y_trn_val, y_trn_val), axis=0)

    else:  # if ds_subset == "train-valid-test":
        X_trn_val, y_trn_val = ds["train-valid"]
        X_tst, y_tst = ds["test"]

        l_ds_Train, l_ds_Valid = len(y_trn_val), len(y_tst)
        print(f"{X_trn_val.shape=} | {y_trn_val.shape=} || {X_tst.shape=} | {y_tst.shape=}")

        X_trn_val = np.concatenate((X_trn_val, X_tst), axis=0)  # size=(L, l_mVos, nC)
        y_trn_val = np.concatenate((y_trn_val, y_tst), axis=0)  # size=(L,)

    X_trn_val = np.moveaxis(X_trn_val, 1, -1)  # size=(L, nC, l_mVos)
    y_trn_val = np.expand_dims(y_trn_val, axis=1)  # size=(L, 1, l_mVos)
    ds_TrnVal = np.hstack((X_trn_val, y_trn_val))

    print(f"{ds_TrnVal.shape=}")  # size=(L, nC+1, l_mVos)
    deets = deets._replace(l_ds_Train=l_ds_Train, l_ds_Valid=l_ds_Valid)
    return ds_TrnVal

def get_tunerXy(Xy, debug=False):
    if debug: print(f"@get_tunerXy: {Xy= }")
    _X = np.moveaxis(Xy[:-1], -1, 0)  # size=(l_mVos, nC)
    _y = int(Xy[-1][0])
    return [_X, _y]

def get_tunerLabel(Xy, debug=False):
    if debug: print(f"@get_tunerLabel: {Xy= }")
    _y = int(Xy[-1][0])
    return vocab_DHG1428[args.n_classes][_y]

def get_splits(ds_TrnVal, debug=False):
    if debug: print(f"@get_splits{ds_TrnVal[:3]= }")
    rand_idxs_TrnVal = L(list(torch.randperm(deets.l_ds_Train).numpy()))
    idxs_Tst = L(range(deets.l_ds_Train, deets.l_ds_Train+deets.l_ds_Valid))
    return rand_idxs_TrnVal, idxs_Tst        
 
def ensembleTunerDataloader(ds_subset, img_size=380, bs=16, return_dls=True, preview=False):
    assert ds_subset in ["train-valid-test", "train-valid", "test"]
    
    tunerDHG1428 = DataBlock(
        blocks=(tunerImageBlock, CategoryBlock),
        get_items=get_tunerPredsTargs,
        get_x=get_tunerXy,
        get_y=get_tunerLabel,
        splitter=get_splits,
        item_tfms=Resize(img_size, method=ResizeMethod.Squish),
        batch_tfms=[Normalize.from_stats(*imagenet_stats)],
    )

    ds = tunerDHG1428.datasets(ds_subset, verbose=False)
    if return_dls:
        dls = tunerDHG1428.dataloaders(ds_subset, bs=bs, worker_init_fn=e_seed_worker, generator=e_repr_gen, device=defaults.device)
        clear_output(wait=False)
        assert dls.c == args.n_classes, ">> ValueError: dls.c != n_classes as specified!!"

        if preview:
            print(dedent(f"""
            Dataloader has been created successfully...
            The dataloader has {len(dls.vocab)} ({dls.c}) classes: {dls.vocab}
            Training set [len={len(dls.train.items)}] loaded on device: {dls.train.device}
            Validation set [len={len(dls.valid.items)}] loaded on device: {dls.valid.device}
            Previewing loaded data [1] and applied transforms [2]...
            """))
            dls.show_batch(nrows=1, ncols=4, unique=False, figsize=(12, 12))
            dls.show_batch(nrows=1, ncols=4, unique=True, figsize=(12, 12))
        else: clear_output(wait=False)

        return dls

    else: return ds
# -----------------------------------------------


def _compare(this, that):
    return round(sum(this == that) / len(that), ndigits=4)

def _ensemble_similarity(dX):
    l_ensemble = L(dX.shape)[1]
    _dX_0 = np.column_stack(L(dX[:, 0])*l_ensemble)
    return round(sum(np.all(dX == _dX_0, axis=1)) / len(dX), ndigits=4)

def _pt_finder(suffix):
    for f in Path(deets.ds_directory).iterdir():
        if f.is_file() and f.stem.endswith(suffix):
            return f

    raise FileNotFoundError(f".pt file with {suffix=} not found!")

def loadEnsembleTunerDataset(mVOs, checkpoint=False, verbose=False):
    ds = {}
    ds_subsets = ["train-valid", "valid", "test"]

    for _subset in ds_subsets:
        X, dX, y = [], [], []

        for _vo in mVOs:
            _pt_file = _pt_finder(suffix=f"{args.n_classes}G-{translateMVOs(_vo)}-[{_subset}]")
            _X, _y = torch.load(f=_pt_file)
            X.append(_X) ; y.append(_y) 

            _dX = np.argmax(_X, axis=2)
            if L(_dX.shape)[1] == 1:
                _similarity = _accuracy = _compare(_dX[:, 0], _y)
                dX.append(np.squeeze(_dX))
            else:  # if L(_dX.shape)[1] == 2:
                _similarity = _compare(_dX[:, 0], _dX[:, 1])
                _accuracy = [_compare(_dX[:, 0], _y), _compare(_dX[:, 1], _y)]
                dX.append(_dX[:, 0]) ; dX.append(_dX[:, 1])

            if verbose:
                print(f"{translateMVOs(_vo):>10}-[{_subset}]: {_X.shape=} | {_accuracy=:.4f}")
        
        # source: https://stackoverflow.com/a/37777691
        _identical_targs = (np.diff(np.vstack(y).reshape(len(y), -1), axis=0) == 0).all()
        assert _identical_targs, "`y` contains different `targs` arrays"

        X = np.concatenate(X, axis=1)
        y = np.column_stack(y)
        dX = np.column_stack(dX)
        ds[_subset] = (X, y)

        if verbose:
            print(f">> {X.shape=} | {y.shape=} | {dX.shape=}")
            print(f">> {_identical_targs=} | {_ensemble_similarity(dX)=:.4f}\n")

    if checkpoint:
        torch.save(obj=ds, f=f"{deets.ds_directory}/eTDs-{args.n_classes}G-{translateMVOs(mVOs)}.pt")
    
    return ds
# -----------------------------------------------


def initTraining(query_tuple, nd, bs, img_sz, eps=None, pct_start=None, log=False):
    learn, e_epochs, e_lr, e_accuracy = query_tuple

    if eps is None: _eps_a = _eps_b = opt_InitEpochs[nd]
    else:           _eps_a, _eps_b = eps if type(eps) is list else [eps, eps]
    print(f"Batch Size: {bs}, Image Size: {img_sz}, Epochs: [{_eps_a}, {_eps_b}]")

    if pct_start is None: _ps_a, _ps_b = opt_pctStart[nd][_eps_a], opt_pctStart[nd][_eps_b]
    else:                 _ps_a, _ps_b = pct_start if type(pct_start) is list else [pct_start, pct_start]

    learn.freeze() ; _tag = f"i{img_sz}{isFrozen(learn)}"
    return_tuple = FitFlatCosine(
        learn, 
        i_tag=_tag, i_eps=_eps_a, i_pct_start=_ps_a, 
        e_epochs_lr_accuracy=(e_epochs, e_lr, e_accuracy),
        log=log,
    )
    learn, e_epochs, e_lr, e_accuracy = return_tuple

    learn.unfreeze() ; _tag = f"i{img_sz}{isFrozen(learn)}"
    return_tuple = FitFlatCosine(
        learn, 
        i_tag=_tag, i_eps=_eps_b, i_pct_start=_ps_b, 
        e_epochs_lr_accuracy=(e_epochs, e_lr, e_accuracy),
        log=log,
    )
    learn, e_epochs, e_lr, e_accuracy = return_tuple

    if hasattr(learn.model, "tuner_img_sz"): 
        return_tuple = FitFlatCosine(
            learn, 
            i_tag=_tag, i_eps=(_eps_b//2), i_pct_start=_ps_b, 
            e_epochs_lr_accuracy=(e_epochs, e_lr, e_accuracy),
            finetune=True, log=log,
        )
        # return_tuple = FitFineTune(
        #     learn, i_tag=f"i{img_sz}f", i_eps=(max(1, int(0.15*_eps_b)), max(1, int(0.35*_eps_b))), 
        #     i_pct_start=_ps_b, e_epochs_lr_accuracy=(e_epochs, e_lr, e_accuracy), log=log, )
        learn, e_epochs, e_lr, e_accuracy = return_tuple

    i_Timer(stt_time=deets.e_stt_datetime)
    return (learn, e_epochs, e_lr, e_accuracy)

def iterativeScaling(query_tuple, nd, bs, img_sz, eps, pct_start=None, finetune=False):
    learn, e_epochs, e_lr, e_accuracy = query_tuple

    if hasattr(learn.model, "tuner_img_sz"): learn.model._change_tuner_img_sz(size=img_sz)
    learn.dls = multiOrientationDataLoader(deets.ds_directory, bs=args.bs, img_size=img_sz, e2eTunerMode=hasattr(learn.model, "tuner_img_sz"))
    print(f"Batch Size: {bs}, Image Size: {img_sz}, Epochs: {eps}")

    if pct_start is None:
        if eps in [3, 5, 10, 15, 20]: _ps = opt_pctStart[nd][eps]
        else: raise Exception("ValueError: valid `eps` options are [3, 5, 10, 15, 20]")
    else: _ps = pct_start

    learn.freeze() ; _tag = f"{img_sz}{isFrozen(learn)}"
    return_tuple = FitFlatCosine(
        learn, 
        i_tag=_tag, i_eps=eps, i_pct_start=_ps, 
        e_epochs_lr_accuracy=(e_epochs, e_lr, e_accuracy),
    )
    learn, e_epochs, e_lr, e_accuracy = return_tuple

    learn.unfreeze() ; _tag = f"{img_sz}{isFrozen(learn)}"
    return_tuple = FitFlatCosine(
        learn, 
        i_tag=_tag, i_eps=eps, i_pct_start=_ps, 
        e_epochs_lr_accuracy=(e_epochs, e_lr, e_accuracy),
    )
    learn, e_epochs, e_lr, e_accuracy = return_tuple

    if finetune: 
        return_tuple = FitFlatCosine(
            learn, 
            i_tag=_tag, i_eps=eps, i_pct_start=_ps, 
            e_epochs_lr_accuracy=(e_epochs, e_lr, e_accuracy),
            finetune=True,
        )
        learn, e_epochs, e_lr, e_accuracy = return_tuple
    
    i_Timer(stt_time=deets.e_stt_datetime)
    return (learn, e_epochs, e_lr, e_accuracy)

def ensembleTunerTraining(query_tuple, bs, img_sz, itr_eps, pct_start=0.5):
    learn, e_epochs, e_lr, e_accuracy = query_tuple
    if not(hasattr(args, "e_history")): setattr(args, "e_history", [])

    l_itr_eps = len(itr_eps)
    assert l_itr_eps%2 == 0, "ValueError: `itr_eps` must be a flat list of successive freeze/unfreeze training rounds"
    itr_eps = [itr_eps[i:i+2] for i in range(0, l_itr_eps, 2)]

    for (_eps_a, _eps_b) in itr_eps:
        print(f"Batch Size: {bs}, Image Size: {img_sz}, Epochs: [{_eps_a}, {_eps_b}]")

        learn.freeze() ; _tag = f"t{img_sz}{isFrozen(learn)} x{_eps_a:02}"
        return_tuple = FitFlatCosine(
            learn, 
            i_tag=_tag, i_eps=_eps_a, i_pct_start=pct_start, 
            e_epochs_lr_accuracy=(e_epochs, e_lr, e_accuracy),
        )
        learn, e_epochs, e_lr, e_accuracy = return_tuple
        args.e_history.extend(learn.recorder.values)
        i_Logger(_tag, evaluateLearner(learn=learn, dls=learn.dls))  # i_Logger(_tag, e_accuracy)

        learn.unfreeze() ; _tag = f"t{img_sz}{isFrozen(learn)} x{_eps_b:02}"
        return_tuple = FitFlatCosine(
            learn, 
            i_tag=_tag, i_eps=_eps_b, i_pct_start=pct_start, 
            e_epochs_lr_accuracy=(e_epochs, e_lr, e_accuracy),
        )
        learn, e_epochs, e_lr, e_accuracy = return_tuple
        args.e_history.extend(learn.recorder.values)
        i_Logger(_tag, evaluateLearner(learn=learn, dls=learn.dls))  # i_Logger(_tag, e_accuracy)

    i_Timer(stt_time=deets.e_stt_datetime)
    return (learn, e_epochs, e_lr, e_accuracy)
# -----------------------------------------------


def evaluateLearner(learn, dls, verbose=False):
    interp = ClassificationInterpretation.from_learner(learn=learn, dl=dls.valid)
    arg_preds = np.argmax(interp.preds, axis=1)
    accuracy = np.round(sum(interp.targs == arg_preds) / len(interp.targs), decimals=4)
    
    clear_output(wait=False)
    if verbose: print(f"@evaluateLearner: learn.model {accuracy=:.4f}")
    return accuracy

def ensembleTunerEvaluation(learn, dls, mVOs):
    train_accuracy = evaluateLearner(learn=learn, dls=dls)
    test_dls = ensembleTunerDataloader(ds_subset="test")
    test_accuracy = evaluateLearner(learn=learn, dls=test_dls)

    _X = test_dls.valid_ds.items[:, :-1, :]
    _y = test_dls.valid_ds.items[:, -1, 0]
    _dX = np.moveaxis(np.argmax(_X, axis=1), -1, 0)

    vo_accuracies = []
    for idx, vo_preds in enumerate(_dX):
        _accuracy = round(sum(vo_preds == _y) / len(_y), ndigits=4)
        vo_accuracies.append(f"{_accuracy:.4f}")
        print(f"{translateMVOs(mVOs[idx]):>10} {_accuracy:.4f}")

    Logger(f"trainAccuracy: {train_accuracy:.4f}~[{'-'.join(vo_accuracies)}]")
    Logger(f"testAccuracy: {test_accuracy:.4f} @{i_Timer(stt_time=e_stt_datetime, stdout=False)}")

def e2eTunerLearnerEvaluation(learn, mVOs, dls=None, ds_idx=1, verbose=False):
    if dls is None: dl = learn.dls[ds_idx].new(shuffled=False, drop_last=False)
    else:           dl = dls[ds_idx]
    
    preds, targs, decoded = learn.get_preds(dl=dl, with_decoded=True)
    tags = [_vo.title() for _vo in mVOs]

    accuracy = [round((sum(i_d == targs) / len(targs)).item(), ndigits=4) for i_d in decoded]
    accuracy = dict(zip([*tags, "Tuner"], accuracy))

    clear_output(wait=False)
    if verbose: Logger(f"accuracies@e2eT: {accuracy}")
    else: return accuracy["Tuner"]
# -----------------------------------------------


class end2endTunerModel(Module):
    def __init__(self, archMultiVOs, archTuner, dls_vocab, tuner_img_sz=224, debug=False):
        self.dls_vocab = dls_vocab
        self.tuner_img_sz = tuner_img_sz
        self.debug = debug

        self.multiVOsBody = create_body(arch=archMultiVOs, cut=None)
        self.multiVOsHead = create_head(nf=num_features_model(self.multiVOsBody), n_out=len(dls_vocab))
        self.tunerBody = create_body(arch=archTuner, cut=None)
        self.tunerHead = create_head(nf=num_features_model(self.tunerBody), n_out=len(dls_vocab))

        self.item_tfms = [Resize(size=tuner_img_sz, method=ResizeMethod.Squish), ToTensor()]
        self.batch_tfms = [IntToFloatTensor(), Normalize.from_stats(*imagenet_stats)]

        self.to(defaults.device)
        if self.debug: print(f"{self= }", end=f"\n{'-'*50}\n\n")

    def forward(self, Xy, y=None):
        if y is not None: X, y = Xy, y  # [NOTE] modified for tensorboard SummaryWriter
        else:             X, y = Xy     # [NOTE] modified for tensorboard SummaryWriter

        if self.debug:
            print(f"\n{type(X)= } | {len(X)= } | {explode_types(X)= }")
            for xi in X: print(f"{explode_types(xi)= } | {xi.shape= }")

        mvo_ftrs = [self.multiVOsBody(xi) for xi in X]
        mvo_ftrs = [self.multiVOsHead(xi) for xi in mvo_ftrs]

        if self.debug:
            for idx in range(len(X)): print(f">> {X[idx].shape= } ~~ {mvo_ftrs[idx].shape= }")

        mvo_preds = torch.cat([torch.unsqueeze(xi.detach(), dim=1) for xi in mvo_ftrs], dim=1)
        tnr_y = self.decode_batch_targs(y)
        tnr_X = self.batch_tuner_images(mvo_preds, tnr_y)
        
        tnr_ftrs = self.tunerBody(tnr_X)
        tnr_ftrs = self.tunerHead(tnr_ftrs)

        if self.debug:
            print(f">> {tnr_X.shape= } | {tnr_y.shape= }")
            print(f">> {mvo_preds.shape= } | {tnr_ftrs.shape= }")
            self.debug = False #; os._exit(os.EX_OK)

        return [*mvo_ftrs, tnr_ftrs]

    def decode_batch_targs(self, yb):
        yb = torch.vstack(yb).moveaxis(0, -1).detach().cpu()
        yb = ["".join(map(chr, _y)).replace(" ", "") for _y in yb]

        vocab = list(self.dls_vocab)
        targs = np.array([vocab.index(_y) for _y in yb])
        
        if self.debug: print(f"\n{yb[:3]= } ~ targs= {targs.tolist()}")
        return targs

    def batch_tuner_images(self, preds, targs):
        batch = []
        preds = preds.detach().cpu().numpy()

        for _idx, (_X, _y) in enumerate(zip(preds, targs)):
            img = tunerImages.create(Xy=[_X, _y])
            img = compose_tfms(img, self.item_tfms)

            if self.debug and (_idx == 0): print(f">> {img.shape= }") ; img.show()  # type:ignore
            batch.append(torch.unsqueeze(img, dim=0))

        batch = torch.cat(batch, dim=0).to(defaults.device)
        batch = compose_tfms(batch, self.batch_tfms)

        if self.debug: print(f"\n{type(batch)= } | {batch.shape= }")
        return batch

    def _change_tuner_img_sz(self, size):
        self.item_tfms[0] = Resize(size=size, method=ResizeMethod.Squish)

    def splitter(self, model:Module):
        bodies = params(model.multiVOsBody) + params(model.tunerBody)
        heads  = params(model.multiVOsHead) + params(model.tunerHead)
        return [bodies, heads]
# -----------------------------------------------


def _object_directory(obj, filter=""):
    return [i for i in dir(obj) if not(i.startswith('__')) and (filter in i)]

def _create_mvo_metric(idx:int):
    exec(dedent(
        f"""

        def accuracyMultiVOs_{idx}(self, preds, targs): 
            return accuracy(preds[{idx}], targs)
            
        """
    ))
    return locals()[f"accuracyMultiVOs_{idx}"]

def attachMetrics(lossWrapper, mVOs, rename=False):
    for _idx, _vo in enumerate(mVOs):
        _vo = _vo.title().replace("-", "")
        setattr(lossWrapper, f"accuracyMultiVOs_{_idx}", _create_mvo_metric(idx=_idx))
        if rename: exec(f"lossWrapper.accuracyMultiVOs_{_idx}.__name__ = 'accuracy{_vo}'")

def returnMetrics(lossFunction, mVOs, verbose=False):
    metrics = []
    for _idx in range(len(mVOs)):
        exec(f"metrics.append(lossFunction.accuracyMultiVOs_{_idx})")
    metrics.append(lossFunction.accuracyTuner)

    if verbose: print(f"\n{metrics= }")
    return metrics

class e2eTunerLossWrapper(Module):
    def __init__(self, nOutputs, baseLoss=CrossEntropyLossFlat):
        self.baseLoss = baseLoss
        self.log_vars = nn.parameter.Parameter(torch.zeros((nOutputs)))
        self.to(defaults.device)

    def forward(self, preds:list, targs):
        mvos_losses = 0
        for _idx, _preds in enumerate(preds[:-1]):
            _mvo_loss = self.baseLoss()(_preds, targs)
            _mvo_precision = torch.exp(-self.log_vars[_idx])
            _mvo_loss = _mvo_precision*_mvo_loss + self.log_vars[1]
            mvos_losses += _mvo_loss

        tnr_loss = self.baseLoss()(preds[-1], targs)
        tnr_precision = torch.exp(-self.log_vars[-1])
        tnr_loss = tnr_precision*tnr_loss + self.log_vars[-1]
        
        return mvos_losses+tnr_loss

    def decodes(self, preds): return [_preds.argmax(dim=-1) for _preds in preds]

    def accuracyTuner(self, preds, targs): return accuracy(preds[-1], targs)
# -----------------------------------------------


@patch
def show_results(self:Learner, ds_idx=1, decode_idx=-1, dl=None, max_n=3, shuffle=True, **kwargs):
    if dl is None: dl = self.dls[ds_idx].new(shuffle=shuffle)
    b = dl.one_batch()
    preds, targs, decoded = self.get_preds(dl=[b], with_decoded=True)

    if type(decoded) == list:
        self.dls.show_results(b, decoded[decode_idx], max_n=max_n, **kwargs)  # type:ignore
    elif type(decoded) == torch.Tensor:
        self.dls.show_results(b, decoded, max_n=max_n, **kwargs)
    else:
        print(">> ValueError: unrecognized dype returned from `learn.get_preds(..., with_decoded=True)`")
# -----------------------------------------------


def generateModelGraph(model, dls, tag="e2eEnsembleTuner"):
    m_tag = f"[{deets.e_secret}]-{tag}-model-{args.n_classes}G.{e_desc_mVOs(args.mv_orientations)}"
    writer = SummaryWriter(log_dir=f"../runs/expsModels/{m_tag}")

    if hasattr(args, "architecture"):
        e_architecture = f"{args.architecture= }"
    else:
        e_architecture = f"""
            {args.mvo_architecture= }
            {args.tnr_architecture= }
        """

    if hasattr(args, "init_img_sz"): init_img_sz = f"{args.init_img_sz= }"
    elif hasattr(args, "tuner_img_sz"): init_img_sz = f"{args.tuner_img_sz= }"
    else: init_img_sz = "args.init_img_sz/args.tuner_img_sz= None"
    
    m_info = f"""
        {m_tag= }
        {args.mv_orientations= }
        {args.bs= }
        {args.n_classes= }
        {e_architecture}
        {init_img_sz}
        {deets.ds_directory= }
        {deets.e_strftime= }
    """
    writer.add_text(tag="model information", text_string=m_info)

    writer.add_graph(model, input_to_model=dls.one_batch()[0], verbose=False)
    # try:   writer.add_graph(model, input_to_model=dls.one_batch()[0], verbose=False)
    # except Exception as e: 
    #     if "CUDA" in str(e): 
    #         stats = os.popen("gpustat -p").readlines(); 
    #         print("".join(stats[0] + stats[args.idx_gpu+1]))

    writer.close()
    print(f">> Model graph generated successfully @{m_tag= }")
    os._exit(os.EX_OK)
# -----------------------------------------------


# -----------------------------------------------
# -----------------------------------------------
class tensorTunerDataset(torch.utils.data.Dataset):
    def __init__(self, ds_subset, no_splits=False, is_valid=False, do_softmax=False, verbose=False):
        assert ds_subset in ["train-valid", "test"]
        self.set_X_y(ds_subset, no_splits, is_valid, do_softmax, verbose)

    def __getitem__(self, idx): return self.X[idx], self.y[idx]
        
    def __len__(self): return len(self.X)

    def set_X_y(self, ds_subset, no_splits, is_valid, do_softmax, verbose):
        self.ds = loadEnsembleTunerDataset(mVOs=args.mv_orientations, verbose=verbose)
        self.vocab = vocab_DHG1428[args.n_classes]

        if ds_subset == "test":
            self.X, self.y = self.ds["test"]

        elif ds_subset == "train-valid":
            if no_splits:
                self.X, self.y = self.ds["train-valid"]

            elif not (no_splits):
                X, y = self.ds["train-valid"]
                X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=deets.e_repr_seed,)

                if is_valid:         self.X, self.y = X_valid, y_valid
                elif not (is_valid): self.X, self.y = X_train, y_train

        assert all(self.y[:, 0] == self.y[:, 1])
        self.y = torch.tensor(self.y[:, 0]).long()
        self.X = torch.tensor(self.X).float()

        if do_softmax: self.X = NN.Softmax(dim=-1)(self.X)
        if args.tuner_network_type == "CNN":
            self.X = torch.cat([self.X]*args.n_classes, dim=-1).reshape((*self.X.shape, -1))

def tensorTunerDataloader(verbose=False):
    if args.tuner_train_subset == "train-valid":
        train_ds = tensorTunerDataset(ds_subset="train-valid", is_valid=False)
        valid_ds = tensorTunerDataset(ds_subset="train-valid", is_valid=True)
        test_ds  = tensorTunerDataset(ds_subset="test", verbose=verbose)

    else:  # elif args.tuner_train_subset == "train-valid-test":
        train_ds = tensorTunerDataset(ds_subset="train-valid", no_splits=True)
        valid_ds = test_ds = tensorTunerDataset(ds_subset="test", verbose=verbose)

    dls = DataLoaders.from_dsets(train_ds, valid_ds, bs=args.bs).to(defaults.device)
    test_dls = DataLoaders.from_dsets(test_ds, test_ds, bs=args.bs).to(defaults.device)

    clear_output(wait=False)
    if verbose:
        print(dedent(f"""
            {dls.train.X.shape=} | {dls.train.y.shape=}
            {dls.valid.X.shape=} | {dls.valid.y.shape=}
            {test_dls.train.X.shape=} | {test_dls.train.y.shape=}
            {test_dls.valid.X.shape=} | {test_dls.valid.y.shape=}
        """))

    return dls, test_dls


class tensorTunerModel_CNN(Module):
    def __init__(self, architecture, n_in, n_out, verbose=False):
        self.encoder = create_body(arch=architecture, n_in=n_in)
        self.head = create_head(nf=num_features_model(m=self.encoder), n_out=n_out)
        
        apply_init(self.head, NN.init.kaiming_normal_)
        self.to(defaults.device)
        if verbose: print(f"{self= }")

    def forward(self, X): return self.head(self.encoder(X))

    def splitter(self, model:Module): return [params(model.encoder), params(model.head)]

class tensorTunerModel_FCN(Module):
    def __init__(self, n_in, n_out, layer_sizes=None, verbose=False):
        self.verbose = verbose
        layer_sizes = [512, 1024, 2048, 4096][::-1] if (layer_sizes is None) else layer_sizes
        self.tuner = self._fastaiHead(n_in, n_out, ps=0.75, momentum=0.25, lin_ftrs=layer_sizes)

        apply_init(self.tuner, NN.init.kaiming_normal_)
        self.to(defaults.device)
        if self.verbose: print(f"{self= }")

    def forward(self, X):
        return self.tuner(X)

    def _fastaiHead(self, n_in, n_out, lin_ftrs, ps=0.5, momentum=0.01):
        lin_ftrs = [n_in] + lin_ftrs
        l_lin_ftrs = len(lin_ftrs)
        ps = L(round(ps*i/l_lin_ftrs, ndigits=2) for i in range(1, l_lin_ftrs+1))
        if self.verbose: print(f"{l_lin_ftrs= } \n{lin_ftrs= } \n{ps= }")

        layers = [
            NN.Flatten(),
            NN.BatchNorm1d(num_features=n_in, momentum=momentum),
            NN.Dropout(p=ps[0]),
        ]
        for _ni, _no, _ps, in zip(lin_ftrs[:-1], lin_ftrs[1:], ps[1:]):
            layers += self._Lin_Bn_Drop(n_in=_ni, n_out=_no, ps=_ps, momentum=momentum)
        layers.append(NN.Linear(in_features=lin_ftrs[-1], out_features=n_out))

        return NN.Sequential(*layers)

    def _Lin_Bn_Drop(self, n_in, n_out, ps, momentum):
        return [
            NN.Linear(in_features=n_in, out_features=n_out),
            NN.ReLU(inplace=True),
            NN.BatchNorm1d(num_features=n_out, momentum=momentum),
            NN.Dropout(p=ps),
        ]

    def splitter(self, model:Module): return [params(model.tuner)]


def tensorTunerTraining(query_tuple, itr_eps, pct_start=0.5):
    learn, e_epochs, e_lr, e_accuracy = query_tuple
    if not(hasattr(args, "e_history")): setattr(args, "e_history", [])

    for _eps in itr_eps:
        print(f"Epochs: [{_eps}, {_eps}]")

        learn.freeze() ; _tag = f"tta.x{_eps:02}{isFrozen(learn)}"
        return_tuple = FitFlatCosine(
            learn, 
            i_tag=_tag, i_eps=_eps, i_pct_start=pct_start, 
            e_epochs_lr_accuracy=(e_epochs, e_lr, e_accuracy),
        )
        learn, e_epochs, e_lr, e_accuracy = return_tuple
        args.e_history.extend(learn.recorder.values)
        i_Logger(_tag, evaluateLearner(learn=learn, dls=learn.dls))

        learn.unfreeze() ; _tag = f"ttb.x{_eps:02}{isFrozen(learn)}"
        return_tuple = FitFlatCosine(
            learn, 
            i_tag=_tag, i_eps=_eps, i_pct_start=pct_start, 
            e_epochs_lr_accuracy=(e_epochs, e_lr, e_accuracy),
        )
        learn, e_epochs, e_lr, e_accuracy = return_tuple
        args.e_history.extend(learn.recorder.values)
        i_Logger(_tag, evaluateLearner(learn=learn, dls=learn.dls))

    i_Timer(stt_time=deets.e_stt_datetime)
    return (learn, e_epochs, e_lr, e_accuracy)

def tensorTunerEvaluation(learn, dls, test_dls):
    train_accuracy = evaluateLearner(learn=learn, dls=dls)
    test_accuracy = evaluateLearner(learn=learn, dls=test_dls)

    if args.tuner_network_type == "CNN":
        _X = test_dls.valid_ds.X[:, :, 0, :].numpy()
    else:  # elif args.tuner_network_type == "FCN":
        _X = test_dls.valid_ds.X.numpy()
    
    _y = test_dls.valid_ds.y.numpy()
    _dX = np.moveaxis(np.argmax(_X, axis=-1), -1, 0)
    print(f"{len(_y)= }")

    vo_accuracies = []
    for idx, vo_preds in enumerate(_dX):
        _accuracy = round(sum(vo_preds == _y) / len(_y), ndigits=4)
        vo_accuracies.append(f"{_accuracy:.4f}")
        print(f"{translateMVOs(args.mv_orientations[idx]):>10} {_accuracy:.4f}")

    Logger(f"trainAccuracy: {train_accuracy:.4f}~[{'-'.join(vo_accuracies)}]")
    Logger(f"testAccuracy: {test_accuracy:.4f} @{i_Timer(stt_time=e_stt_datetime, stdout=False)}")
# -----------------------------------------------
# -----------------------------------------------
