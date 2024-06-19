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
import subprocess

video_data = st.file_uploader("Upload file", ['mp4','mov', 'avi'])

temp_file_to_save = './temp_file_1.mp4'
temp_file_result  = './temp_file_2.mp4'

# func to save BytesIO on a drive
def write_bytesio_to_file(filename, bytesio):
    """
    Write the contents of the given BytesIO to a file.
    Creates the file or overwrites the file if it does
    not exist yet. 
    """
    with open(filename, "wb") as outfile:
        # Copy the BytesIO stream to the output file
        outfile.write(bytesio.getbuffer())

if video_data:
    # save uploaded video to disc
    write_bytesio_to_file(temp_file_to_save, video_data)

    # read it with cv2.VideoCapture(), 
    # so now we can process it with OpenCV functions
    cap = cv2.VideoCapture(temp_file_to_save)

    # grab some parameters of video to use them for writing a new, processed video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_fps = cap.get(cv2.CAP_PROP_FPS)  ##<< No need for an int
    st.write(width, height, frame_fps)
    
    # specify a writer to write a processed video to a disk frame by frame
    fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')
    out_mp4 = cv2.VideoWriter(temp_file_result, fourcc_mp4, frame_fps, (width, height),isColor = False)
   
    while True:
        ret,frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) ##<< Generates a grayscale (thus only one 2d-array)
        out_mp4.write(gray)
    
    ## Close video files
    out_mp4.release()
    cap.release()

    ## Reencodes video to H264 using ffmpeg
    ##  It calls ffmpeg back in a terminal so it fill fail without ffmpeg installed
    ##  ... and will probably fail in streamlit cloud
    convertedVideo = "./testh264.mp4"
    subprocess.call(args=f"ffmpeg -y -i {temp_file_result} -c:v libx264 {convertedVideo}".split(" "))
    
    ## Show results
    col1,col2 = st.columns(2)
    col1.header("Original Video")
    col1.video(temp_file_to_save)
    col2.header("Output from OpenCV (MPEG-4)")
    col2.video(temp_file_result)
    col2.header("After conversion to H264")
    col2.video(convertedVideo)
