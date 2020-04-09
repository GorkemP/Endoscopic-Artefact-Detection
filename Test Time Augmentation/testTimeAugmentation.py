import abc
from abc import ABC

#abstract class
class IPredictor(ABC):
    #constructor
    def __init__(self, weightPath):
        self.pathPesos = weightPath

    @abc.abstractmethod
    def predict(self,imgPath):
        pass

#heritage
class DarknetYoloPred(IPredictor):
    
    def __init__(self,weightPath,fichNames, fichCfg):
        IPredictor.__init__(self, weightPath)
        self.fichNames = fichNames
        self.fichCfg = fichCfg

    def predict(self, imgPath, output, conf):
        import detect
        detect.mainDataset(imgPath, output, conf, self.pathPesos, self.fichNames, self.fichCfg)


class MXnetYoloPred(IPredictor):

    def __init__(self,weightPath,classes):
        IPredictor.__init__(self, weightPath)
        self.classes=classes

    def predict(self, imgPath, output, conf):
        import predict_batch
        predict_batch.mainDataset(imgPath, output, conf,'yolo3_darknet53_custom', self.pathPesos, self.classes)

class MXnetSSD512Pred(IPredictor):

    def __init__(self,weightPath,classes):
        IPredictor.__init__(self, weightPath)
        self.classes=classes

    def predict(self, imgPath, output, conf):
        import predict_batch
        predict_batch.mainDataset(imgPath, output, conf,'ssd_512_resnet50_v1_custom',self.pathPesos, self.classes)

class MXnetFasterRCNNPred(IPredictor):
    
    def __init__(self,weightPath,classes):
        IPredictor.__init__(self, weightPath)
        self.classes=classes

    def predict(self, imgPath, output, conf):
        import predict_batch
        predict_batch.mainDataset(imgPath, output, conf,'faster_rcnn_resnet50_v1b_custom', self.pathPesos, self.classes)

class RetinaNetResnet50Pred(IPredictor):
    
    def __init__(self,weightPath,classes):
        IPredictor.__init__(self, weightPath)
        self.classes=classes

    def predict(self, imgPath, output, conf):
        import predict_batch_retinanet
        predict_batch_retinanet.mainDataset(imgPath, output, conf,'resnet50_v1', self.pathPesos, self.classes)

class MaskRCNNPred(IPredictor):
    
    def __init__(self,weightPath,classes):
        IPredictor.__init__(self, weightPath)
        self.classes=classes

    def predict(self, imgPath, output, conf):
        import predict_batch_rcnn
        predict_batch_rcnn.mainDataset(imgPath, output, conf, self.pathPesos, self.classes)


class Efficient(IPredictor):

    def __init__(self, weightPath, classes):
        IPredictor.__init__(self, weightPath)
        self.classes = classes

    def predict(self, imgPath, output, conf):
        import predict_batch_efficient
        predict_batch_efficient.mainDataset(imgPath, output, conf, self.pathPesos, self.classes)

class FSAF(IPredictor):

    def __init__(self, weightPath, classes):
        IPredictor.__init__(self, weightPath)
        self.classes = classes

    def predict(self, imgPath, output, conf):
        import predict_batch_FSAF
        predict_batch_FSAF.mainDataset(imgPath, output, conf, self.pathPesos, self.classes)

class FCOS(IPredictor):

    def __init__(self, weightPath, classes):
        IPredictor.__init__(self, weightPath)
        self.classes = classes

    def predict(self, imgPath, output, conf):
        import predict_batch_FCOS
        predict_batch_FCOS.mainDataset(imgPath, output, conf, self.pathPesos, self.classes)
        
class custom_retinanet(IPredictor):
    # Some basic setup:
    # Setup detectron2 logger
    import detectron2
    from detectron2.utils.logger import setup_logger
    setup_logger()

    # import some common libraries
    import numpy as np
    import glob
    import os
    import cv2
    import random
    from google.colab.patches import cv2_imshow

    # import some common detectron2 utilities
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import MetadataCatalog
    from detectron2.structures import BoxMode

    from detectron2.config import get_cfg

    class_names = ["specularity", "saturation", "artifact", "blur", "contrast", 
                   "bubbles", "instrument", "blood"]

    model_path = "/home/ws2080/Desktop/codes/detectron/model_retinanet/output_32"

    cfg = get_cfg()

    cfg.merge_from_file(model_path+"/config.yaml")
    cfg.MODEL.WEIGHTS = os.path.join(model_path, "model_0059999.pth")
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.1   

    cfg.DATASETS.TEST = ("ead_validation_1",)

    predictor = DefaultPredictor(cfg)

    VALIDATION_PATH = r'/home/ws2080/Desktop/data/training/test/'
    
    def __init__(self):
        IPredictor.__init__(self)

    def predict(self, imgPath, output, conf):
        paths = glob.glob(imgPath+"*.jpg")
        
        for path in paths:
            im = cv2.imread(path)
            outputs = predictor(im)
            
            jpg_file_name = path.split("/")[-1]
            txt_file_name = jpg_file_name.replace(".jpg", ".txt")    

            total_detection = len(outputs["instances"])

            temp_detection_list = []
            detections = outputs["instances"]

            for i in range(total_detection):
                temp_detection = class_names[int(detections.pred_classes[i])]+" "+ str(float(detections.scores[i]))+" "+str(float(detections.pred_boxes.tensor[i,0]))+" "+ str(float(detections.pred_boxes.tensor[i,1])) +" "+str(float(detections.pred_boxes.tensor[i,2]))+" "+ str(float(detections.pred_boxes.tensor[i,3]))
                temp_detection_list.append(temp_detection)

            with open(txt_file_name, 'w') as f:
                    for item in temp_detection_list:
                        f.write("%s\n" % item)       

        
