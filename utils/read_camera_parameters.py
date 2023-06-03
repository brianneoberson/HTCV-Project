import json
import io
import numpy as np
import torch

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
    t = torch.stack(t, dim=0)
    
    return K, R, t