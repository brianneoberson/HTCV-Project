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
- Extract the frames: (<u>TO-DO</u>: replace with our extraction method)
```
./scripts/extractAll.sh 170307_dance1/
```
<u>TO-DO</u>: add steps for segmentation
## Create the python environment for this project:
Create a conda environment with the packages from requirements.txt
```
conda create -n htcv
conda activate htcv
conda install pip
pip install requirements.txt
```

<u>Note</u>: if we install a new package in the environement, refresh the requirements.txt file:
```
pip freeze > requirements.txt
```