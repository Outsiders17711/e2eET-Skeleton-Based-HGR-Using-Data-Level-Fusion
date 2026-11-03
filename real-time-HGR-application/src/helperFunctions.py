# e2eET Skeleton Based HGR Using Data-Level Fusion
# pyright: reportGeneralTypeIssues=false
# pyright: reportOptionalMemberAccess=false
# pyright: reportWildcardImportFromLibrary=false
# -----------------------------------------------
import re
from collections import namedtuple


__all__ = [
    "e_desc_mVOs",
    "translateMVOs",
    "multiDetailsParser",
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
        mVOs = [re.split(delimiters, mVOs)]
    elif (type(mVOs) == list) and (type(mVOs[0]) == str):
        if (len(mVOs) >= 2): mVOs = [re.split(delimiters, i) for i in mVOs]
        elif (len(mVOs) < 2): mVOs = [mVOs]

    translation = "_".join([".".join([translation_map[vo] for vo in _mvo_]) for _mvo_ in mVOs])
    if debug: print(f"@translateMVOs: correction={mVOs} >> translation=[{translation}]")
    return f"[{translation}]"


multiDetailsParser = namedtuple(
    typename="multiDetailsParser",
    field_names=[
        "e_desc", "e_tag", "e_secret", "e_model_tag",
        "e_repr_seed", "e_stt_datetime", "e_strftime", "e_details",
        "learn_directory", "ds_directory"
    ],
    defaults=[
        None, None, None, None,
        None, None, None, None,
        None, None
    ]
)


def hgrLogger(*args, log, end="\n"):
    with open(log, "a") as f: print(*args, end=end, file=f)
    print(*args, end=end)
