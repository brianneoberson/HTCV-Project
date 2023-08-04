import json
import io
import numpy as np
import torch

def reshape_camera_matrices(K, R, T):
    K_ = torch.zeros(K.shape[0], 4, 4)
    K_[:,0:3,0:3] = K
    K_[:,3,3] = 1
    return K_, R, T


def read_camera_parameters(filename):
    f = open(filename, "r")
    data = json.load(f)
    
    K, R, t = [], [], []
    
    for camera in data['cameras']:
        if camera['type'] == 'hd':
            K.append(torch.from_numpy(np.array(camera['K'])))
            R.append(torch.from_numpy(np.array(camera['R'])))
            t.append(torch.from_numpy(np.array(camera['t'])))
    
    K = torch.stack(K, dim=0)
    R = torch.stack(R, dim=0)
    t = torch.squeeze(torch.stack(t, dim=0))
    
    return K, R, t