import cv2
camera = cv2.VideoCapture(0,cv2.CAP_DSHOW)
face_casade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
while True:
    check , frame = camera.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_casade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)
    for x,y,w,h in faces:
        frame=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),thickness=3)
    cv2.imshow("face",frame)
    key=cv2.waitKey(1)
    if key==ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

