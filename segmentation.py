import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import os

predictor = None

def load_predictor():
    global predictor
    if predictor is None:
        try:
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
            cfg.MODEL.DEVICE = "cpu"
            predictor = DefaultPredictor(cfg)
            print("Detectron2 predictor loaded successfully.")
        except Exception as e:
            print(f"Error loading Detectron2: {e}")
    return predictor

def perform_segmentation(image_path):
    load_predictor()
    if predictor is None:
        return {"error": "Detectron2 predictor not loaded."}
    try:
        image = cv2.imread(image_path)
        if image is None:
            return {"error": f"Failed to load image: {image_path}"}
        outputs = predictor(image)
        instances = outputs["instances"].to("cpu")
        results = {
            "classes": instances.pred_classes.tolist(),
            "scores": instances.scores.tolist(),
            "masks": instances.pred_masks.tolist() if instances.has("pred_masks") else []
        }
        return results
    except Exception as e:
        return {"error": f"Segmentation failed: {str(e)}"}

load_predictor()
