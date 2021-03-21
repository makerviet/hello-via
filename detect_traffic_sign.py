import cv2
from src.traffic_sign_detection.traffic_sign_detector import TrafficSignDetector

model = TrafficSignDetector("models/via_traffic_sign_detection_20210321.pt", use_gpu=True)

# define a video capture object 
vid = cv2.VideoCapture(0) 
  
while(True): 
      
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read()

    preds, viz_img = model.predict(frame, visualize=True)
  
    # Display the resulting frame 
    cv2.imshow('viz_img', viz_img) 
      
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  
# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 