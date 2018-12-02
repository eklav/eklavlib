
import cv2, time
face_cascade = cv2.CascadeClassifier("D:/NEWAGEPROJECTS/facerecognition/eklavlibfolder/haarcascade_frontalface_alt.xml")

def image_facerecog(imgread):
        #gray_img = cv2.cvtColor(imgread, cv2.COLOR_BGR2GRAY)

    imgread = reshapeimagetobelow1000(imgread)
    faces = face_cascade.detectMultiScale(imgread, scaleFactor = 1.05, minNeighbors = 5)
    img_detected = imgread #incase face is not detected, return not a nothing, rather a zero. else called before referenced error
    for x,y,w,h in faces:
        img_detected = cv2.rectangle(imgread, (x,y), (x+w, y+h), (0,255,0),3)
    return img_detected

#videovar = cv2.VideoCapture("D:/NEWAGEPROJECTS/facerecognition/facerecog_fromvideo/trimfinal.mp4")
def videorecord (cameranumber,facedetectornot) :
    videovar = cv2.VideoCapture(cameranumber)
    #videovar = cv2.VideoCapture(0) #capturing the video starts, but it is not stored.
    while True: #this loop will break if "q" is entered
        check, frame = videovar.read() #store each frame in a variable frame
        if facedetectornot :
            graywithbox = image_facerecog(frame)
            cv2.imshow('each frame',graywithbox)
        else :
            cv2.imshow('each frame', frame)
        key = cv2.waitKey(1) #loop repeats after 1ms
        if key==ord('q') :
            break
    videovar.release()
    cv2.destroyAllWindows()

#videorecord(0, True)
####################################
def reshapeimagetobelow1000 (img) :
    while True :
        if img.shape[0]>1000 and img.shape[1]>1000 : #and means if either falls below 1000 leave looping. not useful for vv diff w and h photos
           img = cv2.resize(img, (int(img.shape[1]/2),int(img.shape[0]/2)))
        else :
            break
    return img
####################################
