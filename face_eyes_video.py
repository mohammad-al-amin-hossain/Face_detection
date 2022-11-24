import cv2 as cv

def face_detect(frame, face_detector, eyes_detector):

    f_S = frame

    f_b = cv.cvtColor(f_S, cv.COLOR_BGR2GRAY)

    face_detected = face_detector.detectMultiScale(f_b, scaleFactor = 1.1, minNeighbors = 5)
    eyes_detected = eyes_detector.detectMultiScale(f_b, scaleFactor = 1.1, minNeighbors = 5)
    
    for (x, y, w, h) in eyes_detected:
        cv.rectangle(f_S, (x, y), (x + w, y + h), (0, 0, 250), 2)

    n_o_f = len(face_detected)
    count = n_o_f
    if count:
        for x, y, w, h in face_detected:
            cv.rectangle(f_S, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv.putText(img=f_S, text=f'face {count}', org=(x, y), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1,
                       color=(255, 0, 0), thickness=1)
            count -= 1


    cv.putText(img=f_S, text=f'Face Count = {n_o_f}', org=(50, 50), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=2,
               color=(0, 0, 0), thickness=2)
    return f_S


def read_video(src):
    capture = cv.VideoCapture(src)
    face_detector = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
    eyes_detector = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_eye.xml")
    while True:
        isTrue, frame = capture.read()
        f = face_detect(frame, face_detector, eyes_detector)
        cv.imshow('Face Detected', f)
        if cv.waitKey(10) & 0xFF == ord('q'):
            break
    capture.release()
    cv.destroyAllWindows()

#read_video(src="/Users/mohammadalaminhossain/Downloads/screen-capture.mp4", scale=scale)
read_video(src=0)
