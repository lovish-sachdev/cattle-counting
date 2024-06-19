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
import tempfile
import os

def process_video(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    st.write(total_frames)
    # Create a progress bar
    st_frame=st.empty()
    progress_bar = st.progress(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Example transformation: flipping the frame horizontally
        transformed_frame = cv2.flip(frame, 0)
        
        # If you are running an ML model, it would go here
        # transformed_frame = your_ml_model(frame)

        frame_count += 1
        
        # Update the progress bar
        progress_bar.progress(frame_count / total_frames)
        if frame_count%100==0:
            st.image(transformed_frame,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )
    cap.release()

    progress_bar.empty()  # Clear the progress bar

st.title("Video Transformer")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file is not None:
    with st.spinner('Processing video...'):
        # Create a temporary directory to save the uploaded video
        with tempfile.NamedTemporaryFile(delete=False) as temp_video_file:
            temp_video_file.write(uploaded_file.read())
            temp_video_file_path = temp_video_file.name

        # Define the output video file path
        output_video_path = os.path.join(tempfile.gettempdir(), 'transformed_video.mp4')

        # Process the video with the transformation
        process_video(temp_video_file_path, output_video_path)

