import scipy.io as sio
import numpy as np
mat = sio.loadmat("mpii_human_pose_v1_u12_1.mat")
print(mat['RELEASE'].shape)
print(mat['RELEASE'][0])
print(mat['RELEASE'][0])