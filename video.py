import cv2
import numpy as np
from tensorflow.keras.preprocessing import image 
from keras.models import load_model
from tensorflow.keras.models import model_from_json  
DIR = './results'
img_counter = 0
img_name = ""

# model = model_from_json(open("fer.json", "r").read())  

# #load weights  
model=load_model('fer_final.h5')  


face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def livevideo():
    cam = cv2.VideoCapture(0)

    cv2.namedWindow("test")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("test", frame)

        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            # SPACE pressed
            # img_name = "image_{}.png".format(time.time())
            # completeName = os.path.join("./", img_name)
            # cv2.imwrite(completeName, frame)
            # print("{} written!".format(completeName))
            gray_img= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
        
            faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)  
            
        
            for (x,y,w,h) in faces_detected:
                print('WORKING')
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),thickness=7)  
                roi_gray=gray_img[y:y+w,x:x+h]          #cropping region of interest i.e. face area from  image  
                roi_gray=cv2.resize(roi_gray,(48,48))  
                img_pixels = image.img_to_array(roi_gray)  
                img_pixels = np.expand_dims(img_pixels, axis = 0)  
                img_pixels /= 255
                predictions = model.predict(img_pixels)  
        
                #find max indexed array  
                
                max_index = np.argmax(predictions[0])
                if (max_index==0 or max_index==1 or max_index==2 or max_index==3 or max_index==4 or max_index==5 or max_index==6):
                    print(max_index)
                    cam.release()
                    return max_index

    cv2.destroyAllWindows()