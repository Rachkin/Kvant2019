import cv2
import numpy as np
import torch
import time

emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

model = HarukaNet()
model.load_state_dict(torch.load('my_models/HN_c32_PP_r5.pt'))
model.eval()

cap = cv2.VideoCapture(0)



c_stat = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}

log = ''

log2= ''

s_time = int(time.time())
c_time = int(time.time()) 

p_time = s_time

p_em = 0

p_same_time = s_time

p_stat = 0

file = open("emotion_log.txt","w+")

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
        prediction = model.predict(torch.from_numpy(cropped_img.astype("float32")))
        cv2.putText(frame, emotion_dict[int(np.argmax(prediction))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        
        c_stat[int(np.argmax(prediction))] += 1

    cv2.imshow('frame', frame)
    
    if p_time != c_time:
        c_em = 0
        for i in range(0, 7):
            if(c_stat[c_em] < c_stat[i]):
                c_em = i
                
        if c_stat[c_em] == 0:
            c_em = 7
        
        #   c_log = str(int(c_time - s_time) // 60) + ':' + str((c_time - s_time) % 60) + ' - '
        #  if c_stat[c_em] != 0:
        #      c_log += str(emotion_dict[c_em])
        #  else:
        #      c_log += 'Undefined'
        #  c_log += '\n'
        #  log += c_log
        #  print(c_stat)
        #  print(c_log)
        #print(c_em)
        if p_em != c_em:
            
            c_log2 = '[' + str(int(p_same_time - s_time)//60) + ':' + str(int(p_same_time - s_time)%60) + ' - ' + str(int(c_time - s_time - 1)//60) + ':' + str(int(c_time - s_time - 1)%60) + ']; ';
            if p_em != 7:
                c_log2 += str(emotion_dict[p_em])
            else:
                c_log2 += 'Undefined'
            #if not ((p_same_time == c_time + 1) and (c_time == s_time + 1)):   
            
            c_log2 += '\n'
            
            print(c_log2)
            
            log2 += c_log2
            
            p_same_time = c_time
            
            p_em = c_em
        
        c_stat = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    p_time = c_time
    c_time = int(time.time()) 
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
c_log2 = '[' + str(int(p_same_time - s_time)//60) + ':' + str(int(p_same_time - s_time)%60) + ' - ' + str(int(c_time - s_time - 1)//60) + ':' + str(int(c_time - s_time - 1)%60) + ']; ';

if p_em != 7:
    c_log2 += str(emotion_dict[p_em])
else:
    c_log2 += 'Undefined'
log2 += c_log2
print(c_log2)

file.write(log2)

file.close()

cap.release()
cv2.destroyAllWindows()