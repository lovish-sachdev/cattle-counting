from flask import Flask, request, jsonify
import numpy as np
import os
import onnxruntime as rt
import cv2
from PIL import Image
import io
import base64
from helper import apply_transformation,preprocess,postprocess,get_image,draw_boxes

app = Flask(__name__)

main_dir=os.path.dirname(os.path.abspath(__file__))
onnx_model_path=os.path.join(main_dir,"models","model1"+".onnx")
ort_session = rt.InferenceSession(onnx_model_path)

@app.route('/get_dimensions', methods=['POST'])
def get_count():
    
    if 'image' not in request.files:
        return jsonify({"error": "No image file found"}), 400

    file = request.files['image']
    try:

#         # Read image data as NumPy array
        image = Image.open(file.stream)

#         # Handle potential errors (e.g., invalid image format)
        if image is None:
            return jsonify({'error': 'Failed to decode image'}), 400

        image=np.array(image)
        image=apply_transformation(ort_session,image)
        _, encoded_image = cv2.imencode('.jpg', image)

        # Convert encoded image to bytes and then to base64 string
        image_bytes = encoded_image.tostring()
        base64_string = base64.b64encode(image_bytes).decode('utf-8')

        return jsonify({'image_64': base64_string})

    except Exception as e:
        return jsonify({'error': str(e)}), 500  # Internal Server Error

if __name__ == '__main__':
    app.run(debug=True)  # Set debug=False for production



