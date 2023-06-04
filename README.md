# HTCV-Project
Hot Topics in Computer Vision Project: Neural Representations Shape-from-Silhouette

## Dataset
We use the data provided by [PanopticStudio](http://domedb.perception.cs.cmu.edu/). Here are the steps to prepare the data for our model:

- Clone the [PanopticStudio repository]( https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox)
- Choose the scene from their [list of scenes](https://docs.google.com/spreadsheets/d/1eoe74dHRtoMVVFLKCTJkAtF8zqxAnoo2Nt15CYYvHEE/edit#gid=1333444170), for example take the scene named "dance1".
- Download the scene:
```
cd panoptic-toolbox
./scripts/getData.sh 170307_dance1 0 3 // (this downloads 0 VGA and 3 HD videos)
```
- Extract the frames: (*TO-DO*: replace with our extraction method)
```
./scripts/extractAll.sh 170307_dance1/
```
## Segmentation part structure:
This component of the project comprises a Python script, run.py, and an 'input' folder. The 'input' folder should contain the images you wish to segment.
The run.py script executes image segmentation using the People Segmentation using UNet model (version: 0.0.4).
The segmented images, represented in binary format with a black background and a white figure, will be output into the 'opt' folder.
### Requirements:
Before running this project, make sure you meet the following system requirements:

- Python 3.7 or later
- Additional requirements can be found in the [requirements.txt](/HTCV-Project/Segmentation/people_segmentation-master/requirements.txt) file.

### Segmentation steps:
To run the Image Segmentation script, follow these steps:

1. Clone the repository to your local machine:

```
git clone https://github.com/brianneoberson/HTCV-Project.git
```
2. Navigate to the project directory:
```
cd /HTCV-Project/Segmentation/people_segmentation-master/
```
3. Ensure your images are in the 'input' folder
4. Run the Segmentation.py script:
```
python run.py
```
5. The segmented images will be found in the 'opt' folder.

## Create the python environment for this project:
Create a conda environment from the requirements.txt file
```
conda create -n htcv
conda activate htcv
conda install pip
pip install requirements.txt
```

*Note*: if we install a new package in the environement, refresh the requirements.txt file:
```
pip freeze > requirements.txt
```
## Reference :
cff-version: 1.2.0
message: "If you use this software, please cite it as below."
authors:
  - family-names: Vladimir
    given-names: Iglovikov
    orcid: https://orcid.org/0000-0003-2946-5525
title: "People Segmentation using UNet"
version: 0.0.4
doi: 10.5281/zenodo.7708627
date-released: 2020-10-14
url: https://github.com/ternaus/people_segmentation
