## Implementation of the framework proposed for the Endoscopic Artefact Detection and Segmentation (EAD2020) Challenge - Detection Task

<p align="center">
  <img src="https://github.com/GorkemP/Endoscopic-Artefact-Detection/blob/master/images/EAD2020_frameOnly_01111.jpg">
</p>

Challenge: https://ead2020.grand-challenge.org/Home/

Our method ranked **first** in the detection task.  
First, it is better to read the paper: (*coming soon*) to understand general framework.

A docker environment with installed pyTorch and detectron2 environments can be obtained [here](https://hub.docker.com/repository/docker/splendor90/detectron2)

Object detection models (Faster RCNN, Cascade RCNN and RetinaNet) used in this work are built upon [detectron2 API](https://github.com/facebookresearch/detectron2)

**Ensemble** and **Test Time Augmentation** is adapted from the [Ensemble Methods for Object Detection](https://github.com/ancasag/ensembleObjectDetection)


