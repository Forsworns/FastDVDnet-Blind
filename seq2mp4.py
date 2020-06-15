# use this script to generate the mp4s from the denoised seqs

import skvideo.io
import numpy as np
import os

DIR = './results'
dirs = os.listdir(DIR)
dirs = filter(lambda x: os.path.isdir("{}/{}".format(DIR, x)), dirs)

for d in dirs:
    idx = d.lstrip('mg_test_').rstrip('_damage')
    seqs = os.listdir("{}/{}".format(DIR, d))
    seqs = list(filter(lambda x: x.endswith('.png'), seqs))
    print("processing ", d)
    frame0 = skvideo.io.vread("{}/{}/{}".format(DIR, d, seqs[0]))
    _, M, N, C = frame0.shape
    F = len(seqs)
    v = np.zeros((F, M, N, C), dtype=np.uint8)
    for s in range(len(seqs)):
        v[s] = skvideo.io.vread("{}/{}/n25_FastDVDnet_{}.png".format(DIR, d, s))
    skvideo.io.vwrite("{}/mg_refine_{}.mp4".format(DIR, idx), v)
