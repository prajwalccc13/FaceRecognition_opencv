import numpy as np
import cv2

from utils import CFEVideoConf, image_resize

cap = cv2.VideoCapture(0)

save_path           = 'saved-media/glasses_and_stash.mp4'
frames_per_seconds  = 24
config              = CFEVideoConf(cap, filepath=save_path, res='720p')
out                 = cv2.VideoWriter(save_path, config.video_type, frames_per_seconds, config.dims)
face_cascade        = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eyes_cascade        = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
# smile_cascade       = cv2.CascadeClassifier('E:\\opencv\\test\\cascades\\data\\haarcascade_smile.xml')
# nose_cascade        = cv2.CascadeClassifier('E:\opencv\test\cascades\third-party\Nose18x15.xml')
glasses             = cv2.imread("images/fun/lenses.png", -1)
# mustache            = cv2.imread('images/fun/mustache.png',-1)
# smile_logo          = cv2.imread('images/fun/smile.png',-1)



while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray        = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces       = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    frame       = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w] #rec
        roi_color = frame[y:y+h, x:x+w]
        # cv2.rectangle(frame, (x,y), (x+w, y+h), (255,255,255), 2)

        eyes = eyes_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=5)
        for(ex, ey, ew, eh) in eyes:
            # cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)
            roi_eyes = roi_gray[ey:ey+eh, ex:ex+ew]
            glasses2 = image_resize(glasses.copy(), width=ew)

            gw, gh, gc = glasses2.shape
            for i in range(0, gw):
                for j in range(0, gh):
                    #print(glasses[i, j]) #RGBA
                    if glasses2[i, j][3] != 0: # alpha 0
                        roi_color[ey + i, ex + j] = glasses2[i, j]


        # smile = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=5)
        # for(sx, sy, sw, sh) in smile:
        #     # cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)
        #     roi_eyes = roi_gray[sy:sy+sh, sx:sx+sw]
        #     smile2 = image_resize(smile_logo.copy(), width=sw)

        #     ssw, ssh = smile2.shape
        #     for i in range(0, ssw):
        #         for j in range(0, ssh):
        #             # if smile2[i, j][3] != 0: # alpha 0
        #                 roi_color[sy + i, sx + j] = smile2[i, j]


        # nose = nose_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=5)
        # for(nx, ny, nw, nh) in eyes:
        #     cv2.rectangle(frame, (nx,ny), (nx+nw, ny+h), (255,0,0), 2)
        #     roi_eyes = roi_gray[ny:ny+nh, nx:nx+nw]

        
    frame       = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    # Display the resulting frame
    out.write(frame)
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()