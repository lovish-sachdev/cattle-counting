import requests
import base64
import cv2
import numpy as np
API_URL = "http://127.0.0.1:5000/get_dimensions"


# Path to the image you want to send
image_path = r"C:\Users\07032\github_projects\for_shadaab\cattle_image.jpg"

# Open the image in binary mode
with open(image_path, 'rb') as image_file:
    files = {'image': (image_file.name, image_file)}

    # Send a POST request with the image file
    response = requests.post(API_URL, files=files)

# Check for successful response
if response.status_code == 200:
    print("YES")
    # # Parse the JSON response
    data = response.json()
    image_64 = data.get("image_64")
    print(f"Image dimensions:image_64 = {type(image_64)}")
    image_bytes = base64.b64decode(image_64)
    # Decode bytes into NumPy array using cv2.imdecode
    image_array = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    cv2.imshow("image",image_array)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
else:
    print(f"Error: {response.status_code}")
    print(response.text)  # Optional: Print the error message from the API