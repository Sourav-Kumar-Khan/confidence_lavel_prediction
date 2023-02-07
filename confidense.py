import tensorflow as tf
model = tf.keras.models.load_model(r"E:\SproutsAI\image_sentiment_detection\confidense.h5")
import cv2
import numpy as np
import os
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
import mediapipe as mp

# load face detection model
mp_face = mp.solutions.face_detection.FaceDetection(
    model_selection=1, # model selection
    min_detection_confidence=0.5 # confidence threshold
)


font = cv2.FONT_HERSHEY_SIMPLEX 
# org
org = (20, 30)

# fontScale
fontScale = 0.7

# Blue color in BGR
color = (255,0, 0)

# Line thickness of 2 px
thickness = 1

def get_confidense_score(image):
    try:
        #img= cv2.imread(image,cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (48,48))
        img = img/255.0
        pred = model.predict(np.expand_dims(np.expand_dims(img, axis = -1), axis=0))
        pred = pred[0][0]
        return pred*100
    except:
        return "not found"

def detect_face(dframe):
    image_rows, image_cols,_ = dframe.shape
    image_input = cv2.cvtColor(dframe, cv2.COLOR_BGR2RGB)
    results = mp_face.process(image_input)
    try:
        detection=results.detections[0]
        location = detection.location_data
        relative_bounding_box = location.relative_bounding_box
        rect_start_point = _normalized_to_pixel_coordinates(
            relative_bounding_box.xmin, relative_bounding_box.ymin, image_cols,
            image_rows)
        rect_end_point = _normalized_to_pixel_coordinates(
            relative_bounding_box.xmin + relative_bounding_box.width,
            relative_bounding_box.ymin + relative_bounding_box.height, image_cols,
            image_rows)
     
        ## Lets draw a bounding box
        color = (255, 0, 0)
        thickness = 2
        cv2.rectangle(image_input, rect_start_point, rect_end_point, color, thickness)
        xleft,ytop=rect_start_point
        xright,ybot=rect_end_point

        crop_img = image_input[ytop: ybot, xleft: xright]
        return crop_img
    except:
        return "No Face Detected"
    

cap = cv2.VideoCapture(0) #for web cam
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)
target_h = 360
target_w = int(target_h * frame_width / frame_height)


while True:
    success, image = cap.read()
    if success:
        #image = resize_image(image)
        image1 = detect_face(image)
        if image1 == "No Face Detected":
            image = cv2.putText(image, "No Face Detected", org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)
        else:
            result = get_confidense_score(image1)  
            # Using cv2.putText() method
            image = cv2.putText(image, f'Confidence Score :{"%.2f" % result}'+"%", org, font, 
                            fontScale, color, thickness, cv2.LINE_AA)
        cv2.imshow("face",image)
        #out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
    
cap.release()
#out.release()
cv2.destroyAllWindows()
    
 