import json
import io
import numpy as np
import torch

def reshape_camera_matrices(K, R, T):
    K_ = torch.zeros(K.shape[0], 4, 4)
    K_[:,0:3,0:3] = K
    K_[:,3,3] = 1
    return K_, R, T

def get_center_scale(Ts):
    poses = Ts
    center = torch.mean(poses, dim=0)
    scale = 1./float(torch.max(torch.abs(poses)))
    return center, scale

def local_to_world(Rc, C):
    R = torch.transpose(Rc, 0, 1)
    t = torch.matmul(-R, C)
    return R, t

def world_to_local(R, t):
    Rc = torch.transpose(R, 0, 1)
    C = torch.matmul(-Rc, t)
    return Rc, C

def read_camera_parameters(filename):
    f = open(filename, "r")
    data = json.load(f)
    
    Ks, Rs, ts = [], [], []
    
    for camera in data['cameras']:
        if camera['type'] == 'hd':
            R = torch.tensor(camera['R'])
            t = torch.squeeze(torch.tensor(camera['t']))
            Rs.append(R)
            ts.append(t)
            # R.append(torch.from_numpy(np.array(camera['R'])))
            # t.append(torch.from_numpy(np.array(camera['t'])))
            if 'K' in camera:
                Ks.append(torch.from_numpy(np.array(camera['K'])))
    
    if Ks != []:
        Ks = torch.stack(Ks, dim=0)
    Rs = torch.stack(Rs, dim=0)
    ts = torch.squeeze(torch.stack(ts, dim=0))

    return Ks, Rs, ts

def read_camera_parameters_world(filename):
    f = open(filename, "r")
    data = json.load(f)
    
    Ks, Rs, ts = [], [], []
    
    for camera in data['cameras']:
        if camera['type'] == 'hd':
            Rc = torch.tensor(camera['R'])
            C = torch.squeeze(torch.tensor(camera['t']))
            R, t = local_to_world(Rc, C)
            Rs.append(R)
            ts.append(t)
            # R.append(torch.from_numpy(np.array(camera['R'])))
            # t.append(torch.from_numpy(np.array(camera['t'])))
            if 'K' in camera:
                Ks.append(torch.from_numpy(np.array(camera['K'])))
    
    if Ks != []:
        Ks = torch.stack(Ks, dim=0)
    Rs = torch.stack(Rs, dim=0)
    ts = torch.squeeze(torch.stack(ts, dim=0))

    return Ks, Rs, ts

def normalize_cameras(Rs, ts):
    center, scale = get_center_scale(ts)
    ts -= center
    ts *= scale
    Rcs = []
    Cs = []
    for i in range(ts.shape[0]):
        Rc, C = world_to_local(Rs[i], ts[i])
        Rcs.append(Rc)
        Cs.append(C)

    Rcs = torch.stack(Rcs, dim=0)
    Cs = torch.stack(Cs, dim=0)
    return Rcs, Cs