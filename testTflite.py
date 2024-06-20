import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import cv2 
import numpy as np
import tensorflow as tf 

camera = cv2.VideoCapture(0)

interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

while(True):
    try : 

        ret, img = camera.read()
        h, w ,c = img.shape

        frame = cv2.resize(img, (128, 128))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, frame = cv2.threshold(frame, 100, 255, cv2.THRESH_BINARY)
        input = np.expand_dims(frame, 0)
        input = np.expand_dims(input, 3)
        input = input.astype("float32")
        interpreter.set_tensor(input_details[0]['index'], input)
        interpreter.invoke()        
        out1 = interpreter.get_tensor(output_details[0]['index'])
        out2 = interpreter.get_tensor(output_details[1]['index'])



        cls = np.argmax(out1[0])

        if cls != 0 :
            if cls == 1 :
                char = "H"
            if cls == 2 :
                char = "S"
            if cls == 3 :
                char = "U"

            x1 = out2[0][0] * (w / 128) * 128 
            y1 = out2[0][1] * (h / 128) * 128 
            x2 = out2[0][2] * (w / 128) * 128 
            y2 = out2[0][3] * (h / 128) * 128

            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, char, (50, 50), cv2.FONT_HERSHEY_SIMPLEX , 3, (0, 255, 0), 2, cv2.LINE_AA) 


        cv2.imshow("cam" , img)
        cv2.imshow("thresh" , frame)

        cv2.waitKey(1)

    except (KeyboardInterrupt) :
        cv2.destroyAllWindows()
        break





