# HTCV-Project
Hot Topics in Computer Vision Project: Neural Representations Shape-from-Silhouette

## Getting started:
Clone the repository and create a conda environment from the requirements.txt file.
```
git clone https://github.com/brianneoberson/HTCV-Project.git
cd HTCV-Project
```
Make sure you have miniconda or Anaconda installed before you continue.
```
conda create -n htcv
conda activate htcv
conda install pip
pip install -r requirements.txt
conda install -c conda-forge opencv
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip install git+https://github.com/tatsy/torchmcubes.git

```

## Dataset
We use the data provided by [PanopticStudio](http://domedb.perception.cs.cmu.edu/). Here are the steps to prepare the data for our model:

- Choose the scene from their [list of scenes](https://docs.google.com/spreadsheets/d/1eoe74dHRtoMVVFLKCTJkAtF8zqxAnoo2Nt15CYYvHEE/edit#gid=1333444170), for example take the scene named "170307_dance1".
- Download the scene videos (we only use the HD videos):
```
./scripts/getData.sh 170307_dance1 0 31 // (this downloads 0 VGA and all HD videos)
```
The above will save the videos under `data/170307_dance1`.
- Extract the frames:
```
python ./scripts/extract.py --dir data/170307_dance1/hdVideos --output-root data/170307_dance1/extracted_frames
```

Next we need to segment the images to obtain silhouette masks of the objects in the scene. For this we use a pretrained model for people segmentation provided by Vladimir Iglovikov's [repository](https://github.com/ternaus/people_segmentation). All you need to do is run:
```
python ./scripts/segment.py --input_dir data/170307_dance1/extracted_frames
```
The masks will be saved under  `data/170307_dance1/silhouettes`.
## Train
*TO-DO*

## Export Mesh
*TO-DO*

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
