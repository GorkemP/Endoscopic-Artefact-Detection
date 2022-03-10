## Implementation of the framework proposed for the Endoscopic Artefact Detection and Segmentation (EAD2020) Challenge - Detection Task

<p align="center">
  <img src="https://github.com/GorkemP/Endoscopic-Artefact-Detection/blob/master/images/EAD2020_frameOnly_01111.jpg">
</p>

### Challenge: https://ead2020.grand-challenge.org/Home/

#### 🏆 Our method ranked [**first**](https://ead2020.grand-challenge.org/evaluation/results/) in the artefact detection task.

First, it is better to read the paper to understand general framework: 

[Endoscopic Artefact Detection with Ensemble of Deep Neural Networks](http://ceur-ws.org/Vol-2595/endoCV2020_paper_id_10.pdf)

A Docker image with installed PyTorch and Detectron2 environments can be obtained [here](https://hub.docker.com/repository/docker/splendor90/detectron2)

Object detection models (Faster RCNN, Cascade RCNN and RetinaNet) used in this work are built upon [Detectron2 API](https://github.com/facebookresearch/detectron2)

**Ensemble** and **Test Time Augmentation** is adapted from the [Ensemble Methods for Object Detection](https://github.com/ancasag/ensembleObjectDetection)

For citation please use the following BibTeX entry

```BibTeX
@inproceedings{polat2020endoscopic,
author    = {Polat, Gorkem and Sen, Deniz and Inci, Alperen and Temizel, Alptekin},
  title     = {Endoscopic Artefact Detection with Ensemble of Deep Neural Networks
               and False Positive Elimination},
  booktitle = {Proceedings of the 2nd International Workshop and Challenge on Computer
               Vision in Endoscopy, EndoCV@ISBI 2020, Iowa City, Iowa, USA, 3rd April
               2020},
  volume    = {2595},
  publisher = {CEUR-WS.org},
  pages     = {8--12},
  year      = {2020}
}
```
