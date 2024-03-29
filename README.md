# HTCV-Project
Hot Topics in Computer Vision Project: Neural Representations Shape-from-Silhouette

Project by Brianne Oberson, Liam Fogarty, Daniia Vergazova and Beatrice Toscano

## Getting started:
Clone the repository and create a conda environment from the requirements.txt file.
```
git clone https://github.com/brianneoberson/HTCV-Project.git
cd HTCV-Project
```
Make sure you have miniconda or Anaconda installed before you continue. Then run the following to create a conda environment.
```
conda create -n htcv python=3.11.4 -y
conda activate htcv
pip install -r requirements.txt
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip install git+https://github.com/tatsy/torchmcubes.git

```
Note: this code has only been tested on an NVIDIA RTX A6000 with 50GB memory.

## Dataset
### Cow 
As mentioned in our report, we conducted our first experiments using a synthetic cow dataset provided by PyTorch3D. This dataset can be downloaded from this [link](https://drive.google.com/drive/folders/1W6MFpS2FkLUoGqJy4lKFQfj4i1l-9VFi?usp=drive_link).
### Panoptic Studio
We use the data provided by [PanopticStudio](http://domedb.perception.cs.cmu.edu/). Downloading the data takes a long time, as we are downloading 31 videos. We provide a google drive link with one example data already extracted and segmented [here](https://drive.google.com/drive/folders/1W6MFpS2FkLUoGqJy4lKFQfj4i1l-9VFi?usp=drive_link). Here are the steps to download and prepare a scene of your choice:

- Choose the scene from PanopticStudio's [list of scenes](https://docs.google.com/spreadsheets/d/1eoe74dHRtoMVVFLKCTJkAtF8zqxAnoo2Nt15CYYvHEE/edit#gid=1333444170), for example take the scene named "150821_dance285".
- Download the scene videos (we only use the HD videos):
```
sh ./scripts/getData.sh 150821_dance285 0 29 // (this downloads 0 VGA and all HD videos)
```
The above will save the videos under `data/150821_dance285/hdVideos`.
- Extract the frames:
```
python ./scripts/extract.py --dir data/150821_dance285/hdVideos --output-root data/150821_dance285/images
```

Next we need to segment the images to obtain silhouette masks of the objects in the scene. For this we use a pretrained model for people segmentation provided by Vladimir Iglovikov's [repository](https://github.com/ternaus/people_segmentation). All you need to do is run:
```
python ./scripts/segment.py --input_dir data/150821_dance285/images
```
The masks will be saved under  `data/150821_dance285/silhouettes`.

### Custom Data
If you want to use your custom data, make sure to organize the files following this structure: 
```
├── custom_data_name
│   ├── images
│   │   ├── hd_00_00_frame0.jpg
│   │   ├── hd_00_01_frame0.jpg
│   │   ├── ...
│   ├── silhouettes
│   │   ├── hd_00_00_frame0.jpg
│   │   ├── hd_00_01_frame0.jpg
│   │   ├── ...
│   ├── calibration.json
```
And make sure the `calibration.json` file has the same structure as the examples found in the [example datasets](https://drive.google.com/drive/folders/1W6MFpS2FkLUoGqJy4lKFQfj4i1l-9VFi?usp=drive_link).

## Train
For training, a config file should be set up. We provide different config files which we used for the experiments mentioned in the report in the `config` folder, as well as explanation of the necessary parameters. Here we use `cow_sc_stratified.yaml` for example:
```
python main.py --config configs/cow_sc_stratified.yaml 
```
The checkpoints and tensorboard logs will be saved in the `output` folder in a subfolder named after the experiment name scpecified in the config file. 

## Export to Mesh
To export the learned volumetric function to a `.ply` triangle mesh, you can run the follwing:
```
  python export_to_mesh.py --chkpt path/to/checkpoint --output path/to/output/file
```

## Camera Plot
We also provide a script for plotting the camera positions, to make sure the calibration information is as expected. This can be done by running:
```
  python plot_cameras.py --calibration path/to/calibration/file
```
## References :

### Dataset:
```
@inproceedings{Joo_2015_ICCV,
author = {Joo, Hanbyul and Liu, Hao and Tan, Lei and Gui, Lin and Nabbe, Bart and Matthews, Iain and Kanade, Takeo and Nobuhara, Shohei and Sheikh, Yaser},
title = {Panoptic Studio: A Massively Multiview System for Social Motion Capture},
booktitle = {ICCV},
year = {2015} }

@inproceedings{Joo_2017_TPAMI,
title={Panoptic Studio: A Massively Multiview System for Social Interaction Capture},
author={Joo, Hanbyul and Simon, Tomas and Li, Xulong and Liu, Hao and Tan, Lei and Gui, Lin and Banerjee, Sean and Godisart, Timothy Scott and Nabbe, Bart and Matthews, Iain and Kanade, Takeo and Nobuhara, Shohei and Sheikh, Yaser},
journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
year={2017} }

@inproceedings{Simon_2017_CVPR,
title={Hand Keypoint Detection in Single Images using Multiview Bootstrapping},
author={Simon, Tomas and Joo, Hanbyul and Sheikh, Yaser},
journal={CVPR},
year={2017} }

@inproceedings{joo2019ssp,
  title={Towards Social Artificial Intelligence: Nonverbal Social Signal Prediction in A Triadic Interaction},
  author={Joo, Hanbyul and Simon, Tomas and Cikara, Mina and Sheikh, Yaser},
  booktitle={CVPR},
  year={2019}
}
```

### Segmentation:
```
cff-version: 1.2.0
authors:
  family-names: Vladimir
  given-names: Iglovikov
  orcid: https://orcid.org/0000-0003-2946-5525
title: "People Segmentation using UNet"
version: 0.0.4
doi: 10.5281/zenodo.7708627
date-released: 2020-10-14
url: https://github.com/ternaus/people_segmentation
```

### PyTorch3D:
```
@article{ravi2020pytorch3d,
    author = {Nikhila Ravi and Jeremy Reizenstein and David Novotny and Taylor Gordon
                  and Wan-Yen Lo and Justin Johnson and Georgia Gkioxari},
    title = {Accelerating 3D Deep Learning with PyTorch3D},
    journal = {arXiv:2007.08501},
    year = {2020},
}
```
