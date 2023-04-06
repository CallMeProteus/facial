import cv2
import os

# Load the camera feed
cap = cv2.VideoCapture(0)

# Load images from a folder
image_folder = os.getcwd()+'\\images'
images = []
for filename in os.listdir(image_folder):
    img = cv2.imread(os.path.join(image_folder, filename))
    if img is not None:
        images.append(img)

# Match the camera feed with the loaded images
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Crop the face from the camera frame
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Match the face with the loaded images
        for img in images:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=5)
            for (ix, iy, iw, ih) in img_faces:
                img_roi_gray = img_gray[iy:iy+ih, ix:ix+iw]
                img_roi_color = img[iy:iy+ih, ix:ix+iw]

                # Calculate the difference between the camera face and the loaded image face
                diff = cv2.absdiff(roi_gray, img_roi_gray)
                diff_mean = diff.mean()

                # If the difference is below a threshold, consider the face a match and log the name
                if diff_mean < 50:
                    print('Match found: ', filename)

    # Show the camera feed with faces detected
    cv2.imshow('Camera Feed', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
