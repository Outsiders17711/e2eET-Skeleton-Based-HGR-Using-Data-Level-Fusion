# Dynamic Hand Gestures Classification
# pyright: reportGeneralTypeIssues=false
# pyright: reportWildcardImportFromLibrary=false
# ---------------------------------------------------------
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# ---
import secrets
from argparse import ArgumentParser
from fastai.vision.all import *
from _helperFunctions import *


# ---------------------------
ap = ArgumentParser(description="Dynamic Hand Gestures Classification: Live Stream Demo")

# --- required arguments
ap.add_argument("-IG", "--idx_gpu", required=True, type=int, choices=[0, 1, 2, 3])
# --- arguments with default options
ap.add_argument("--n_classes", type=int, choices=[7, 14], default=7)
ap.add_argument("--mv_orientations", type=str, nargs="+", default=["custom", "top-down", "front-away"])
ap.add_argument("--nd", type=str, choices=["3d", "2d"], default="3d")
ap.add_argument("--bs", type=int, default=16)
# --- arguments with aliases
ap.add_argument("-IIS", "--init_img_sz", type=int, default=224)
ap.add_argument("-IE", "--init_eps", type=int, choices=[3, 5, 10, 15, 20, 25, 30, 35, 40], default=20)
ap.add_argument("-IISE", "--init_itr_scl_eps", type=int, choices=[0, 5, 10, 15], default=10)
ap.add_argument("-ISIS", "--itr_scl_sizes", type=int, nargs="+", default=["<OPT>"])
ap.add_argument("-ISE", "--itr_scl_eps", type=int, choices=[3, 5, 10, 15], default=5)
# --- "store_true" >>  `False` by default | "store_false" >>  `True` by default
ap.add_argument("-IFT", "--itr_finetuning", action="store_true")
ap.add_argument("-CETE", "--create_e_tb_events", action="store_false")
ap.add_argument("-CEMC", "--create_e_model_checkpoint", action="store_true")
ap.add_argument("-V", "--verbose", action="store_true")


# [init.experiment.args.info]
# ---------------------------
args = ap.parse_args()

e_desc = f"e2eET.liveStream-DHG1428.3D.{args.n_classes}G-{translateMVOs(args.mv_orientations)}-i{args.init_eps}"
if args.itr_scl_sizes != [0]: e_desc += f".{args.itr_scl_eps}"


# [init.hypervariables]
# ---------------------------
e_tag = os.path.basename(__file__)[:-3]
e_secret = secrets.token_hex(2)
e_model_tag = f"../[{e_secret}]-{e_desc}"  # wrt `learn.path/learn.model_dir`

defaults.device = torch.device("cuda", args.idx_gpu)
torch.cuda.set_device(args.idx_gpu)
e_epochs, e_accuracy, e_lr = 0, 0.0, 0.001

learn_directory = f"./.sources/{e_secret}"
if args.n_classes == 14:
    ds_directory = "../images_d/DHG1428.mVOs-3d.14g-noisy(raw).960px-[allVOs].adaptive-mean"
else:
    ds_directory = "./.sources/DHG1428.liveStream-3D.7G-[custom.topdown.frontaway]"


# [import.functions.classes]
# ---------------------------
deets = multiDetailsParser(
    e_desc=e_desc, e_tag=e_tag, e_secret=e_secret, e_model_tag=e_model_tag,
    learn_directory=learn_directory, ds_directory=ds_directory,
    e_repr_seed=None, e_stt_datetime=None, e_strftime=None, e_details=None,
)
from _functionsClasses import *


# [create.dls.model.learn]
# ---------------------------
dls = multiOrientationDataLoader(deets.ds_directory, bs=args.bs, img_size=args.init_img_sz, preview=True)

model = end2endTunerModel(
    archMultiVOs=models.resnet50,
    archTuner=models.resnet18,
    tuner_img_sz=args.init_img_sz,
)

attachMetrics(e2eTunerLossWrapper, args.mv_orientations, rename=True)
loss_func = e2eTunerLossWrapper(nOutputs=len(args.mv_orientations) + 1)
metrics = returnMetrics(loss_func, args.mv_orientations, verbose=True)
learn = Learner(
    dls, model,
    metrics=metrics, loss_func=loss_func,
    opt_func=Adam, splitter=model.splitter,
    path=deets.learn_directory, model_dir=".",
)


# [initTraining]
# ---------------------------
query_tuple = (learn, e_epochs, e_lr, e_accuracy)
return_tuple = initTraining(
    query_tuple, nd=args.nd, bs=args.bs,
    img_sz=args.init_img_sz, eps=args.init_eps, log=True
)
learn, e_epochs, e_lr, e_accuracy = return_tuple


# [init.iterativeScaling]
# ---------------------------
if args.init_itr_scl_eps:
    e_accuracy = 0.0  # reset to track effect of iterative training
    init_itr_scl_img_sz = itr_scl_sizes.pop(0)

    query_tuple = (learn, e_epochs, e_lr, e_accuracy)
    return_tuple = iterativeScaling(
        query_tuple, nd=args.nd, bs=args.bs,
        img_sz=init_itr_scl_img_sz, eps=args.init_itr_scl_eps,
        finetune=opt_InitItrFinetune[args.nd],
    )
    learn, e_epochs, e_lr, e_accuracy = return_tuple
    i_Logger(f"ii{init_itr_scl_img_sz}", e_accuracy)


# [iterativeScaling]
# ---------------------------
for _img_sz_ in itr_scl_sizes:
    if _img_sz_:
        query_tuple = (learn, e_epochs, e_lr, e_accuracy)
        return_tuple = iterativeScaling(
            query_tuple, nd=args.nd, bs=args.bs,
            img_sz=_img_sz_, eps=args.itr_scl_eps,
            finetune=(True if _img_sz_ == itr_scl_sizes[-1] else args.itr_finetuning),
        )
        learn, e_epochs, e_lr, e_accuracy = return_tuple
        i_Logger(f"{_img_sz_}", e_accuracy)


# [tunerEvaluation.createCheckpoint(?)]
# ---------------------------
learn.load(deets.e_model_tag, with_opt=True)  # ensure that best model is loaded
e2eTunerLearnerEvaluation(learn=learn, mVOs=args.mv_orientations, verbose=True)

if args.create_e_model_checkpoint: modelCheckpoint(learn, learn_directory)
else: os.remove(Path(f"{learn.path}/{learn.model_dir}/{deets.e_model_tag}.pth"))


# [post.processing]
# ---------------------------
Logger(f"e2eT_MaxA: {e_accuracy:.4f} @{i_Timer(stt_time=e_stt_datetime, stdout=False)}")
print(f"Training completed @{e_tag} ->> [{e_secret}]-{e_strftime} ->- {e_desc}")
Cleaner(target=learn.path)
