import os 

this_file_dir = os.path.dirname(os.path.realpath(__file__))

configs = [
    f"{this_file_dir}/configs/dance_sc_stratified.yaml",
]

for config in configs:

    cmd = f"""
        python main.py --config {config}
    """
    print(cmd)
    os.system(cmd)



