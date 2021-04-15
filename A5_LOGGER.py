# -*- coding: utf-8 -*-
# VideoCapture使用時のpyinstallerでのexe化はsetuptoolsを44.0.0にバージョンダウンする必要あり
# pip uninstall setuptools
# pip install setuptools==44.0.0
# keras==2.2.4, tensorflow==1.15.0

import cv2
import time
import datetime
from keras.models import model_from_json
import numpy as np
import pandas as pd

aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

def save_frame_camera_cycle():
    # Input camera number and interval time(sec)
    camera_num = int(input('Please input camera port. >>'))
    print('========================')
    time_interval = int(input('Please input time interval(sec). >>'))
    print('========================')
    
    cap = cv2.VideoCapture(camera_num)
    
    # Load model
    model = model_from_json(open('recog_val.json').read())
    # Read the weight of model
    model.load_weights('recog_val.hdf5')
    
    while True:
        ret, frame = cap.read()
        cv2.imshow("frame", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        #sizeを取得
        Height, Width = frame.shape[:2]
        # resize
        img = cv2.resize(frame,(int(Width),int(Height)))
        #マーカーを検出
        corners, ids, rejectedImgPoints = aruco.detectMarkers(img, dictionary) 
        
        if len(corners) > 0:
            #print(ids[0])
            #マーカーid=01検出時
            if ids[0] == 1:
                x_AR = int(corners[0][0][3][0])
                y_AR = int(corners[0][0][3][1])
                print(x_AR, y_AR)
                # input cuttin area x:横, y:縦
                # number 1
                x1_l = x_AR + 17
                y1_l = y_AR - 8
                x1_r = x1_l + 30
                y1_r = y1_l + 35
                # number 2
                x2_l = x1_l + 39
                y2_l = y1_l 
                x2_r = x2_l + 30
                y2_r = y2_l + 35
                # number 3
                x3_l = x2_l + 39
                y3_l = y2_l
                x3_r = x3_l + 30
                y3_r = y3_l + 35
                # number 4
                x4_l = x3_l + 39
                y4_l = y3_l 
                x4_r = x4_l + 30
                y4_r = y4_l + 35

                # Cut val_1
                img1 = img[y1_l:y1_r, x1_l:x1_r]
                # Cut val_2
                img2 = img[y2_l:y2_r, x2_l:x2_r]
                # Cut val_3
                img3 = img[y3_l:y3_r, x3_l:x3_r]
                # Cut val_4
                img4 = img[y4_l:y4_r, x4_l:x4_r]
                
                # Prediction
                imgs = [img1, img2, img3, img4]
                val = []
                for img in imgs:
                    # img = cv2.imread(img)
                    # Grayed
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                    # 28*28 processing

                    Xt = []
                    img = cv2.resize(img,(28, 28), cv2.INTER_CUBIC)
                
                    Xt.append(img)
                    Xt = np.array(Xt)/255
                
                    # Predict
                    result = model.predict_classes(Xt)
                
                    val.append(int(result[0]))
                
                val = int('{}{}{}{}0'.format(*val))
                time_now = datetime.datetime.now()
                print('Time: ', time_now)
                print('Value: ', val)
                print('========================')
                
                list_input = [[time_now, val]]
                df = pd.DataFrame(list_input)
                df.to_csv('A5_LOGGER.csv', mode='a', header=False)
                
                time.sleep(time_interval)
            
    cv2.destroyWindow("frame")
    
save_frame_camera_cycle()