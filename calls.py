import os 

this_file_dir = os.path.dirname(os.path.realpath(__file__))

configs = [
    f"{this_file_dir}/configs/cow_uniform_lambda_0.yaml",
    f"{this_file_dir}/configs/cow_complex_stratified.yaml",
    f"{this_file_dir}/configs/cow_stratified_lambda_0.5.yaml",
    f"{this_file_dir}/configs/cow_stratified_lambda_0.yaml",
    f"{this_file_dir}/configs/cow_color.yaml",
]

for config in configs:

    cmd = f"""
        python main.py --config {config}
    """
    print(cmd)
    os.system(cmd)



