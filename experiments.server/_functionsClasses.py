# e2eET Skeleton Based HGR Using Data-Level Fusion
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
# ---
from __main__ import args, deets
from _helperFunctions import *


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

    "tunerImages",
    "end2endTunerModel",
    "e2eTunerLossWrapper",
    "attachMetrics",
    "returnMetrics",
    "e2eTunerLearnerEvaluation",

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
        label = tuple([ord(c) for c in format(fns[0].parent.parent.name, "32")])
        return cls(tuple((imgs, label)))

    def show(self, ctx=None, **kwargs):
        imgs = list(self[0])
        
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
        dls = multiDHG1428.dataloaders(ds_directory, bs=bs, worker_init_fn=e_seed_worker, generator=e_repr_gen, device=defaults.device, shuffle=shuffle, num_workers=0)
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

def modelCheckpoint(learn, learn_directory):
    _stem = f"[{deets.e_secret}]-{args.n_classes}G-{translateMVOs(args.mv_orientations)}"
    _path_mask = f"../checkpoints/{_stem}"
    
    learn = learn.load(file=deets.e_model_tag, with_opt=True)
    learn.export(fname=f"{_path_mask}.pkl")
    print(f"Learner pkl and pt checkpoints created successfully")
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
    writer.close()
    print(f">> Model graph generated successfully @{m_tag= }")
    os._exit(os.EX_OK)
# -----------------------------------------------
