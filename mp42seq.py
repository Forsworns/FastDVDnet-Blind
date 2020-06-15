# use this script to generate the seqs into the fdvdnet

import skvideo.io
import numpy as np
import os

DIR = './data/test_B'
names = os.listdir(DIR)
names = filter(lambda x: x.endswith('.mp4'), names)
print(names)
for n in names:
    v = skvideo.io.vreader("{}/{}".format(DIR, n))
    d = n.rstrip('.mp4')
    print("processing ", d)
    try:
        os.mkdir("{}/{}".format(DIR, d))
    except:
        pass
    for i, frame in enumerate(v):
        skvideo.io.vwrite("{}/{}/{}.png".format(DIR, d, i), frame)
