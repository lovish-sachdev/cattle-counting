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
#         progress_bar = st.progress(0)
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             texty.write(count)
#             count+=1
#             if count%30==0:
                
#                 input_tensor = preprocess(frame,count)
#                 outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: input_tensor})
#                 predictions = postprocess(outputs[0])
#                 frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
#                 frame=draw_boxes(frame, predictions)
#                 frame=cv2.resize(frame,(320,320))
#                 progress_bar.progress(count*100//1977)
#                 stframe.image(frame,
#                    caption='Detected Video',
#                    channels="RGB",
#                    use_column_width=True
#                    )
#             cv2.waitKey(300)


#         cap.release()
# if __name__ == "__main__":
    
#     main_dir=os.path.dirname(os.path.abspath(__file__))
#     choice=st.selectbox("pick model",["model1","model2"])
#     onnx_model_path=os.path.join(main_dir,"models",choice+".onnx")
#     ort_session = rt.InferenceSession(onnx_model_path)
#     main()


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
#         progress_bar = st.progress(0)
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             texty.write(count)
#             count+=1
#             if count%30==0:
                
#                 input_tensor = preprocess(frame,count)
#                 outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: input_tensor})
#                 predictions = postprocess(outputs[0])
#                 frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
#                 frame=draw_boxes(frame, predictions)
#                 frame=cv2.resize(frame,(320,320))
#                 progress_bar.progress(count*100//1977)
#                 stframe.image(frame,
#                    caption='Detected Video',
#                    channels="RGB",
#                    use_column_width=True
#                    )

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
from moviepy.editor import VideoFileClip,ImageSequenceClip
import numpy as np

# Function to process frames (apply transformations)
def process_frame(frame):
    # Example transformation: Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

# Function to process video
def process_video(input_video_path):
    # Read the video
    video_clip = VideoFileClip(input_video_path)
    
    # Create a temporary file for the output video
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_path = temp_output.name

    # Process each frame
    processed_frames = []
    count=0
    for frame in video_clip.iter_frames():
        count+=1
        if count==500:
            break
        processed_frame = process_frame(frame)
        processed_frames.append(processed_frame)

    # Write the processed frames to a new video
    height, width, _ = processed_frames[0].shape

    clip=ImageSequenceClip(processed_frames,fps=30)

    clip.write_videofile(output_path,codec="libx264")

    return output_path

# Streamlit app
st.title('Video Frame Processing')

# Upload video file
uploaded_file = st.file_uploader('Upload a video file', type=['mp4', 'avi', 'mov'])

if uploaded_file is not None:
    # Save uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_input:
        temp_input.write(uploaded_file.read())
        input_video_path = temp_input.name

    # Process the video and get the output path
    output_video_path = process_video(input_video_path)

    # Display the original video
    st.video(input_video_path)

    # Display the processed video
    st.video(output_video_path)

    st.success('Video processing completed!')
