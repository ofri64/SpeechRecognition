import librosa
import glob
import numpy as np
from numpy import inf
from numpy.linalg import norm
from scipy.spatial import distance


def my_euclidean(a, b):
    return distance.euclidean(a.flatten(), b.flatten())


# train_files = glob.glob('train_data/**', recursive=True)
# test_files = glob.glob('test_files/*')

# train_mfcc = {}
# for f in train_files:
#     label = f.split('/')[1]
#     if f.endswith('.wav'):
#         y, sr = librosa.load(f, sr=None)
#         mfcc = librosa.feature.mfcc(y=y, sr=sr)
#         if label in train_mfcc:
#             train_mfcc[f.split('/')[1]].append(mfcc)
#         else:
#             train_mfcc[f.split('/')[1]] = [mfcc]


# test_mfcc = {}
# for f in test_files:
#     y, sr = librosa.load(f, sr=None)
#     mfcc = librosa.feature.mfcc(y=y, sr=sr)
#     test_mfcc[f.split('/')[1]] = mfcc


def onenn_clf(train_mfcc, test_mfcc, dist_metric):
    classifs = {}
    for name, test_vec in test_mfcc.items():
        min_dist = float('inf')
        out = None
        for c, train_vecs in train_mfcc.items():
            for train_vec in train_vecs:
                dist = dist_metric(test_vec, train_vec)
                if dist < min_dist:
                    min_dist = dist
                    out = c
        classifs[name] = out
    return classifs


def dtw_fast(ts1, ts2):
    curr_row = np.full(ts2.shape[1], inf)
    prev_row = curr_row.copy()
    for i in range(ts1.shape[1]):
        for j in range(ts2.shape[1]):
            dist = norm(ts1[:, i]-ts2[:, j])
            exp1 = prev_row[j-1] if (j-1) >= 0 else inf
            exp2 = curr_row[j-1] if (j-1) >= 0 else inf
            exp3 = prev_row[j]
            min_val = min(exp1,exp2,exp3)
            curr_row[j] = dist + (min_val if min_val < inf else 0)
        prev_row[:] = curr_row
    return curr_row[-1]


