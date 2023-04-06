import cv2
import os

def load_known_faces(known_faces_dir, face_cascade):
    known_faces = []
    known_names = []
    # for file in os.listdir(known_faces_dir):
    #     image = cv2.imread(os.path.join(known_faces_dir, file))
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    #     for (x, y, w, h) in faces:
    #         face = gray[y:y+h, x:x+w]
    #         known_faces.append(face)
    #         known_names.append(file.split('.')[0])
    
    # Read from video file and detect faces
    video_file = "jafeth.mp4"
    cap = cv2.VideoCapture(video_file)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            known_faces.append(face)
            known_names.append("jafeth")
    
    cap.release()
    print(known_faces)
    return known_faces, known_names
