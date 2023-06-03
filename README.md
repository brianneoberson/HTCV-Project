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
This component of the project comprises a Python script, Segmentation.py, and an 'input' folder. The 'input' folder should contain the images you wish to segment. The Segmentation.py script executes image segmentation using the Segment Anything model. The segmented images, represented in binary format with a black background and a white figure, will be output into the 'opt' folder. 

### Segmentation steps:
To run the Image Segmentation script, follow these steps:

1. Clone the repository to your local machine:

```
git clone https://github.com/brianneoberson/HTCV-Project.git
```
2. Navigate to the project directory:
```
cd HTCV-Project/Segmentation/segment-anything-main/
```
3. Ensure your images are in the 'input' folder
4. Run the Segmentation.py script:
```
python Segmentation.py
```
5. The segmented images will be found in the 'opt' folder.

## Create the python environment for this project:
Create a conda environment from the requirements.txt file
```
conda create -n htcv
conda activate htcv
conda install pip
pip install -r requirements.txt
conda install -c conda-forge opencv
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip install git+https://github.com/tatsy/torchmcubes.git
pip install trimesh
```

*Note*: if we install a new package in the environement, refresh the requirements.txt file:
```
pip freeze > requirements.txt
```
