import cv2
import datetime

def recognize_faces(frame, face_cascade, face_encodings_dict,lbl,tlbl,listbox,currentuser):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Recognize faces in the frame and display matching faces
    listindex=0
    for (x, y, w, h) in faces:
        # Extract the face region from the frame
        face = gray[y:y+h, x:x+w]

        # Compare the face with the known faces
        match = None
        for name, encoding in face_encodings_dict.items():
            result = cv2.matchTemplate(face, encoding, cv2.TM_CCOEFF_NORMED)
            _, confidence, _, _ = cv2.minMaxLoc(result)
            if confidence > 0.5:#adjust value to increase accuracy
                match = name
                break

        # Display the matching face with a label
        if match is not None:
            cv2.putText(frame, match, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            lbl(match)
            tlbl(str( datetime.datetime.now()))
            listbox(listindex,'Student '+ match+' confirmed '+ 'Arival time: '+ str( datetime.datetime.now()))
            currentuser(frame)
            print('Student '+ match+' confirmed '+ 'Arival time: '+ str( datetime.datetime.now()))
            listindex +=1

    return frame
