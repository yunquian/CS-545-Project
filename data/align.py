from typing import Tuple
import numpy as np

from numba import njit


def _cosine_dist(u: np.ndarray, v: np.ndarray) -> np.float64:
    uv = np.mean(u * v)
    uu = np.mean(np.square(u))
    vv = np.mean(np.square(v))
    denominator = np.sqrt(uu * vv)
    if denominator == 0:
        return np.float64(0)
    dist = 1 - uv / denominator
    return np.abs(dist)


_cosine = njit(_cosine_dist)


def _dtw_mtx(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param a: (n_features, n_samples) numpy array
    :param b: (n_features, n_samples) numpy array
    :return:
    """
    n, m = a.shape[1], b.shape[1]
    dist = np.zeros((n, m), dtype=np.float64)
    dist[0, 0] = _cosine(a[:, 0], b[:, 0])
    last = np.zeros((n, m), dtype=np.int64)
    for i in range(1, n):
        dist[i, 0] = dist[i - 1, 0] + _cosine(a[:, i], b[:, 0])
        last[i, 0] = -1
    for i in range(1, m):
        dist[0, i] = dist[0, i - 1] + _cosine(a[:, 0], b[:, i])
        last[0, i] = 1
    for i in range(1, n):
        for j in range(1, m):
            min_last = dist[i - 1, j - 1]
            if dist[i - 1, j] < min_last:
                min_last = dist[i - 1, j]
                last[i, j] = -1
            if dist[i, j - 1] < min_last:
                min_last = dist[i, j - 1]
                last[i, j] = 1
            dist[i, j] = min_last + _cosine(a[:, i], b[:, j])
    return dist, last


dtw_matrix = njit(_dtw_mtx)


def dtw_align(a, b, selected_frames=None, well_defined=True):
    """
    :param a: (n_features, n_samples) numpy array
    :param b: (n_features, n_samples) numpy array
    :param selected_frames: (arr1, arr2), boolean arrays
    :param well_defined: the alignment would be a well defined function,
        that is, each frame in 'a' will be matched to a unique frame in 'b'
    :return: list of aligned pairs
    Description
    ------
    pairs = dtw_align(spectrogram1, spectrogram2)

    pairs: [(i,j), ...], i is index in spectrogram 1 and j in spectrogram 2
    """
    dist, last = dtw_matrix(a, b)
    assert dist.shape == last.shape
    pts = []
    i, j = last.shape[0], last.shape[1]
    i = i - 1
    j = j - 1
    while i >= 0 and j >= 0:
        pts.append((i, j))
        if last[i, j] == -1:
            i -= 1
        elif last[i, j] == 1:
            j -= 1
        else:
            i -= 1
            j -= 1
    if well_defined:
        f = {}
        for i, j in pts:
            if i not in f or dist[i, j] < dist[i, f[i]]:
                f[i] = j
        pts = [(i, f[i]) for i in f.keys()]
    if selected_frames is not None:
        a_selected, b_selected = selected_frames
        new_pts = []
        for i, j in pts:
            if a_selected[i] and b_selected[j]:
                new_pts.append((i, j))
        pts = new_pts
    pts.sort()
    return pts
