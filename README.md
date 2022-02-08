
# Underwater Image Fusion Enhancement
A python implementation of a fusion enhancement algorithm for underwater images.

I created this implementation as part of Kaggle's [Tensorflow Greate Barrier Reef](https://www.kaggle.com/c/tensorflow-great-barrier-reef) competition.

This implementation is based on the Java implementation from the [OptimizedImageEnhance](https://github.com/IsaacChanghau/OptimizedImageEnhance) repo.

The original algorithm is from [Enhancing underwater images and videos by fusion](https://ieeexplore.ieee.org/document/6247661) by Cosmin Ancuti.


## Installing as a Package
```
python -m pip install git+https://github.com/DamonGeorge/UnderwaterImage-FusionEnhancement.git
```
Feel free to specify a release tag or other commit or marker (eg by appending `@v0.1`).

## Installing for Development
First create a python 3.7 environment.
Next install all dependencies using
```
python -m pip install -r requirements-dev.txt
```
Initialize pre-commit using
```
pre-commit install
```


## Using the library
Basic import example: `import fusion_enhance`

Specified import example: `from fusion_enhance import enhance`


## Using the example script
You can run the example script as
```
python enhance_files.py <example_image>
```
to enhance an image.
This script also supports the batch enhancement of entire folders using multiple processes.
