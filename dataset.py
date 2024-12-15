import h5py

filename = "/home/mc_lmy/workspace/Structformer/data_new_objects/examples_circle_new_objects/result/batch_500000/leonardo/data00500002.h5"
h5 = h5py.File(filename, 'r')
ids = {}
import json 
import numpy as np
goal_specification = json.loads(str(np.array(h5["goal_specification"])))
print(goal_specification)
exit()
for k in h5.keys():
    if k.startswith("id_"):
        print(h5[k][()])
        ids[k[3:]] = h5[k][()]