from ultralytics import YOLO
   import cv2

   # Load YOLOv8 model (pre-trained on COCO)
   model = YOLO("yolov8m-seg.pt")

   def segment_image(image):
       """Segment objects in the image using YOLOv8."""
       results = model(image, verbose=False)
       return results[0].masks.data.cpu().numpy() if results[0].masks is not None 
       else None
