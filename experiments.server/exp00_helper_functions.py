# pyright: reportGeneralTypeIssues=false
# pyright: reportOptionalMemberAccess=false
# pyright: reportWildcardImportFromLibrary=false
# -----------------------------------------------
import re
from collections import namedtuple

__all__ = [
    "e_desc_mVOs",
    "translateMVOs",
    "tunerDetailsParser",
    "multiDetailsParser",
    "even_chunks",
    "schedulerGPU_mVOs",
    "datasetDirectories",
    "datasetOptimalConfigs",
]
# -----------------------------------------------


def e_desc_mVOs(mVOs:list):
    return f"[{'&'.join(mVOs).replace('ssPV', '').replace('-', '')}]"

def translateMVOs(mVOs, debug=False):
    if debug: print(f"@translateMVOs: {mVOs=}, {type(mVOs)=}, {type(mVOs[0])=}, {len(mVOs)=}")
    translation_map = {
        "top-down": "td", "front-to": "ft", "front-away": "fa",
        "side-right": "sr", "side-left": "sl", "custom": "cm",
    }
    delimiters = r"\s|\+|\||_|&|/|\\"

    # [NOTE]: `mVOs` must be a list of lists
    if (type(mVOs) == str):
        # mVOs = [mVOs.split("+")]
        mVOs = [re.split(delimiters, mVOs)]
    elif (type(mVOs) == list) and (type(mVOs[0]) == str):
        # if (len(mVOs) >= 2): mVOs = [i.split("+") for i in mVOs]
        if (len(mVOs) >= 2): mVOs = [re.split(delimiters, i) for i in mVOs]
        elif (len(mVOs) < 2): mVOs = [mVOs]

    translation = "_".join([".".join([translation_map[vo] for vo in _mvo_]) for _mvo_ in mVOs])
    if debug: print(f"@translateMVOs: correction={mVOs} >> translation=[{translation}]")
    return f"[{translation}]"
# -----------------------------------------------


tunerDetailsParser = namedtuple(
    typename="tunerDetailsParser",
    field_names=[
        "e_desc", "e_tag", "e_secret", "e_model_tag",
        "learn_directory", "ds_directory",
        "e_repr_seed", "e_stt_datetime", "e_strftime", "e_details",
        "l_ds_Train", "l_ds_Valid",
    ],
)

multiDetailsParser = namedtuple(
    typename="multiDetailsParser",
    field_names=[
        "e_desc", "e_tag", "e_secret", "e_model_tag",
        "e_repr_seed", "e_stt_datetime", "e_strftime", "e_details",
        "learn_directory", "ds_directory"
    ],
)
# -----------------------------------------------


def lazy_chunks(sequence:list, n:int):
    """
    Yield successive n-sized chunks from sequence (not optimal).
    Reference: https://stackoverflow.com/a/312464
    """
    for i in range(0, len(sequence), n): yield sequence[i:i + n]

def even_chunks(sequence:list, n:int):
    """
    Yield successive n-sized chunks from sequence distributed as evenly as possible.
    Reference: https://stackoverflow.com/a/2135920
    """
    n = min(n, len(sequence)) # don't create empty buckets
    k, m = divmod(len(sequence), n)
    for i in range(n): yield sequence[i*k+min(i, m):(i+1)*k+min(i+1, m)]

def schedulerGPU_mVOs(idx_gpu, viewOrientations, depth, nClasses=[14, 28], nGPUs=4):
    allTasksGPUs = []
    for vo_1 in viewOrientations:
        for nc in nClasses:
            if depth == 2:
                for vo_2 in viewOrientations:
                    if vo_1 != vo_2: allTasksGPUs.append([" ".join([vo_1, vo_2]), nc])
            elif depth == 1:
                allTasksGPUs.append([vo_1, nc])

    chunks = list(even_chunks(sequence=allTasksGPUs, n=nGPUs))

    print(f"\n>> n_tasksGPUs: {len(allTasksGPUs)}  |  taskDistribution: {[len(i) for i in chunks]} ")
    return chunks[idx_gpu]
# -----------------------------------------------


datasetDirectories = {
    "CNR16": "CNR-3d-original-1920px.1080px-[topdown]",

    "DHG142814": "DHG1428.mVOs-3d.14g-noisy(raw).960px-[allVOs].adaptive-mean",
    "DHG142828": "DHG1428.mVOs-3d.28g-noisy(raw).960px-[allVOs].adaptive-mean",

    "FPHA45": "FPHA.mVOs-dictPaperSplit-3d.V1-noisy(raw).960px-[allVOs].adaptive-mean",

    "LMDHG13": "LMDHG.mVOs-dictPaperSplit-3d.V1-noisy(raw).960px-[allVOs].adaptive-mean",

    "SHREC201714": "SHREC2017.mVOs-3d.14g-noisy(raw).960px-[allVOs].adaptive-mean",
    "SHREC201728": "SHREC2017.mVOs-3d.28g-noisy(raw).960px-[allVOs].adaptive-mean",
}


datasetOptimalConfigs = {
    "CNR16": [
        "top-down", 224, 0.9705,
        "[3c20]-allDatasets-mVO.e2eEnsembleTuning-e2eT-CNR.3d-16G-[td]-i30",
    ],

    "DHG142814": [
        "custom top-down front-away", 224, 0.9583,
        "[c52c]-allDatasets-mVO.e2eEnsembleTuning-e2eT-DHG1428.3d-14G-[cm_td_fa]-i25",
    ],
    "DHG142828": [
        "custom top-down front-away", 224, 0.9238,
        "[31be]-allDatasets-mVO.e2eEnsembleTuning-e2eT-DHG1428.3d-28G-[cm_td_fa]-i25",
    ],

    "FPHA45": [
        "front-away custom top-down", 328, 0.9183,
        "[2c1a]-allDatasets-mVO.e2eEnsembleTuning-e2eT-FPHA.3d-45G-[fa_cm_td]-i30.10",
    ],

    "LMDHG13": [
        "custom front-away top-down", 224, 0.9897,
        "[4d15]-allDatasets-mVO.e2eEnsembleTuning-e2eT-LMDHG.3d-13G-[cm_fa_td]-i25.15",
    ],

    "SHREC201714": [
        "front-away custom front-to", 224, 0.9786,
        "[16d8]-allDatasets-mVO.e2eEnsembleTuning-e2eT-SHREC2017.3d-14G-[fa_cm_ft]-i20",
    ],
    "SHREC201728": [
        "front-away custom front-to", 224, 0.9536,
        "[66dd]-allDatasets-mVO.e2eEnsembleTuning-e2eT-SHREC2017.3d-28G-[fa_cm_ft]-i20",
    ],
}
# -----------------------------------------------
