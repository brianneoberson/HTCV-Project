import os 

this_file_dir = os.path.dirname(os.path.realpath(__file__))

cmd_nerf_light = f"""
    python main.py --config {this_file_dir}/configs/nerf_light.yaml
"""
print(cmd_nerf_light)
os.system(cmd_nerf_light)

cmd_nerf_complex = f"""
    python main.py --config {this_file_dir}/configs/nerf_complex.yaml
"""
print(cmd_nerf_complex)
os.system(cmd_nerf_complex)

cmd_nerf_color = f"""
    python main.py --config {this_file_dir}/configs/debug_color.yaml
"""
print(cmd_nerf_color)
os.system(cmd_nerf_color)



