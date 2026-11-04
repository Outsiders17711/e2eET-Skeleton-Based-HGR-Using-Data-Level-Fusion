# e2eET Skeleton Based HGR Using Data-Level Fusion
# pyright: reportGeneralTypeIssues=false
# pyright: reportWildcardImportFromLibrary=false
# ---------------------------------------------------------
import numpy as np


# [SMMOTHENING FUNCTIONS]______________________________________________________
# [REFERENCE]: https://stackoverflow.com/a/63458548
def _np_convolve(array, span):
    return np.convolve(array, np.ones(span * 2 + 1) / (span * 2 + 1), mode="same")


def _custom_np_convolve(array, span):
    convolve_vector = np.convolve(array, np.ones(span * 2 + 1) / (span * 2 + 1), mode="same")
    convolve_vector[0] = np.average(array[:span])

    # The "my_average/custom" part: shrinks the averaging window on the side that reaches
    # beyond the data, keeps the other side the same size as given  by "span"
    for i in range(1, span + 1):
        convolve_vector[i] = np.average(array[: i + span])
        convolve_vector[-i] = np.average(array[-i - span :])

    return convolve_vector


def _np_cumsum(array, span):
    cumsum_vector = np.cumsum(np.insert(array, 0, 0))
    return (cumsum_vector[span:] - cumsum_vector[:-span]) / float(span)


def _custom_np_cumsum(array, span):
    cumsum_vector = np.cumsum(array)
    moving_average = (cumsum_vector[2 * span :] - cumsum_vector[: -2 * span]) / (2 * span)

    # The "my_average/custom" part. Slightly different to before, because the
    # moving average from cumsum is shorter than the input and needs to be padded
    front, back = [np.average(array[:span])], []
    for i in range(1, span):
        front.append(np.average(array[: i + span]))
        back.insert(0, np.average(array[-i - span :]))
    back.insert(0, np.average(array[-2 * span :]))

    return np.concatenate((front, moving_average, back))


# [DATASET DENOISING FUNCTION]_________________________________________________
def denoise_gesture_sequences(ds_sequences, sz_filter=10, n_samples=None, smthn_function=None):
    smthn_function = _custom_np_convolve if smthn_function is None else smthn_function
    ds_s_shape = ds_sequences.shape

    assert len(ds_s_shape) in [3, 4], "input should be a single / an array of gesture sequence(s)"
    if len(ds_s_shape) == 3:
        ds_sequences = np.expand_dims(ds_sequences, axis=0)

    _ds_sequences_ = []
    for idx, s in enumerate(ds_sequences):
        s = s.transpose()
        s_shape = s.shape

        _s_ = []
        for xyz in range(s_shape[0]):
            for lm in range(s_shape[1]):
                _s_.append(smthn_function(s[xyz, lm], sz_filter))

        _ds_sequences_.append(np.array(_s_).reshape(*s_shape[:2], -1).transpose())

    _ds_sequences_ = np.array(_ds_sequences_)

    n_skeletons = _ds_sequences_.shape[-3]
    if n_samples and n_samples < n_skeletons:
        samples = np.linspace(0, n_skeletons, n_samples, endpoint=False, dtype=int)
        _ds_sequences_ = _ds_sequences_[:, samples, :, :]

    return _ds_sequences_.squeeze()
