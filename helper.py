import onnxruntime as rt
import numpy as np
import cv2 
import streamlit as st
import cv2
import tempfile
import os

def get_image(image_path):
    image = cv2.imread(image_path)
    return image

def preprocess(image):
    input_size = (640, 640)  # Replace with your model's input size
    image_resized = cv2.resize(image, input_size)
    image_resized = image_resized.astype(np.float32) / 255.0
    image_transposed = np.transpose(image_resized, (2, 0, 1))
    input_tensor = np.expand_dims(image_transposed, axis=0)
    return input_tensor

def postprocess(outputs, confidence_threshold=0.20, iou_threshold=0.20):
    predictions = outputs[0]  
    boxes, scores, class_ids = [], [], []
    for prediction in predictions:
        if prediction[4] > confidence_threshold:  
            boxes.append(prediction[:4])
            scores.append(prediction[4])
            # class_ids.append(np.argmax(prediction[5:]))
            class_ids.append(0)

    # Apply NMS
    indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_threshold, iou_threshold)
    if len(indices)==0:
        return []
    return [(boxes[i], scores[i], class_ids[i]) for i in indices.flatten()]

def draw_boxes(image, predictions):
    input_size = (640, 640)  
    image = cv2.resize(image, input_size)
    listt=[]    
    for (box, score, class_id) in predictions:
        x, y, w, h = box
        x=int(x-w/2)
        y=int(y-h/2)
        w=int(w)
        h=int(h)

        cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)

    return image

def apply_transformation(ort_session,frame):
        input_tensor = preprocess(frame)
        outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: input_tensor})
        predictions = postprocess(outputs[0])
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame=draw_boxes(frame, predictions)
        frame=cv2.resize(frame,(320,320))
        
        return frame