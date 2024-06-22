import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import cv2 
import numpy as np
from tensorflow.keras.models import load_model

camera = cv2.VideoCapture(0)
model = load_model('model.h5')

while(True):
    try : 

        ret, img = camera.read()
        h, w ,c = img.shape

        img = cv2.resize(img, (128, 128))
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        ret, frame = cv2.threshold(frame, 85, 255, cv2.THRESH_BINARY)  
        input = (np.expand_dims(frame, 0))
        x = model.predict(input, verbose=2)
        cls = np.argmax(x[1][0])

        if cls != 0 :
            if cls == 1 :
                char = "H"
            if cls == 2 :
                char = "S"
            if cls == 3 :
                char = "U"

            x1 = x[0][0][0] * 128 
            y1 = x[0][0][1] * 128 
            x2 = x[0][0][2] * 128 
            y2 = x[0][0][3] * 128

            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, char, (20, 20), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 255, 0), 1, cv2.LINE_AA) 


        cv2.imshow("cam" , img)
        cv2.imshow("thresh" , frame)

        cv2.waitKey(1)

    except (KeyboardInterrupt) :
        cv2.destroyAllWindows()
        break


