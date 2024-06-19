# import onnxruntime as rt
# import numpy as np
# import cv2 
# import streamlit as st
# import cv2
# import tempfile
# import os

# def get_image(image_path):
#     image = cv2.imread(image_path)
#     return image

# def preprocess(image,count):
#     input_size = (640, 640)  # Replace with your model's input size
#     image_resized = cv2.resize(image, input_size)
#     image_resized = image_resized.astype(np.float32) / 255.0
#     image_transposed = np.transpose(image_resized, (2, 0, 1))
#     input_tensor = np.expand_dims(image_transposed, axis=0)
#     return input_tensor

# def postprocess(outputs, confidence_threshold=0.20, iou_threshold=0.20):
#     predictions = outputs[0]  
#     boxes, scores, class_ids = [], [], []
#     for prediction in predictions:
#         if prediction[4] > confidence_threshold:  
#             boxes.append(prediction[:4])
#             scores.append(prediction[4])
#             # class_ids.append(np.argmax(prediction[5:]))
#             class_ids.append(0)

#     # Apply NMS
#     indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_threshold, iou_threshold)
#     if len(indices)==0:
#         return []
#     return [(boxes[i], scores[i], class_ids[i]) for i in indices.flatten()]

# def draw_boxes(image, predictions):
#     input_size = (640, 640)  
#     image = cv2.resize(image, input_size)
#     listt=[]    
#     for (box, score, class_id) in predictions:
#         x, y, w, h = box
#         x=int(x-w/2)
#         y=int(y-h/2)
#         w=int(w)
#         h=int(h)

#         cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)

#     return image

# ## loading model

# def main():
#     st.title("Video Upload and Process with OpenCV")

#     # File uploader
#     uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

#     if uploaded_file is not None:
        
#         tfile = tempfile.NamedTemporaryFile(delete=False)
#         tfile.write(uploaded_file.read())
#         count=0
#         texty=st.empty()
#         stframe=st.empty()
#         cap = cv2.VideoCapture(tfile.name)
#         stframe = st.empty()
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             input_tensor = preprocess(frame,count)
#             outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: input_tensor})
#             texty.write(count)
#             count+=1
#             predictions = postprocess(outputs[0])
#             frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
#             frame=draw_boxes(frame, predictions)
#             frame=cv2.resize(frame,(320,320))
#             stframe.image(frame, channels="RGB")

#         cap.release()
# if __name__ == "__main__":
    
#     main_dir=os.path.dirname(os.path.abspath(__file__))
#     choice=st.selectbox("pick model",["model1","model2"])
#     onnx_model_path=os.path.join(main_dir,"models",choice+".onnx")
#     ort_session = rt.InferenceSession(onnx_model_path)
#     main()

import streamlit as st
import cv2
from PIL import Image

uploaded_video = st.file_uploader("Choose video", type=["mp4", "mov"])
frame_skip = 300 # display every 300 frames

if uploaded_video is not None: # run only when user uploads video
    vid = uploaded_video.name
    with open(vid, mode='wb') as f:
        f.write(uploaded_video.read()) # save video to disk

    st.markdown(f"""
    ### Files
    - {vid}
    """,
    unsafe_allow_html=True) # display file name

    vidcap = cv2.VideoCapture(vid) # load video from disk
    cur_frame = 0
    success = True
    st_image=st.empty()
    while success:
        success, frame = vidcap.read() # get next frame from video
        if cur_frame % frame_skip == 0: # only analyze every n=300 frames
            print('frame: {}'.format(cur_frame)) 
            pil_img = Image.fromarray(frame) # convert opencv frame (with type()==numpy) into PIL Image
            st_image.image(pil_img)
        cur_frame += 1
