import pymeshlab
import argparse
import os
import numpy as np
from PIL import Image


def create_gif(mesh_path):
    # mesh images to give to GIF animator
    images = []
    
    # create a new MeshSet
    ms = pymeshlab.MeshSet()
    
    # load a new mesh in the MeshSet, and sets it as current mesh
    # the path of the mesh can be absolute or relative
    ms.load_new_mesh(mesh_path)
    
    # first center mesh to origin
    ms.compute_matrix_from_translation(traslmethod = 'Center on Scene BBox')
    
    # rotate around axis and save as img
    for angle in np.linspace(0, 370, 10):
        ms.compute_matrix_from_rotation(rotaxis='Z axis', angle=10)

    
    # output new mesh
    ms.save_current_mesh(os.path.dirname(mesh_path) + "_saved.ply")
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh', type=str, required=True, help='Path to .ply file containing the mesh')
    args = parser.parse_args()
    create_gif(args.mesh)

if __name__ == '__main__':
    main()