import cv2

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the video
cap = cv2.VideoCapture('../output_video.mp4')

# Loop through each frame of the video
while True:
    # Read a frame from the video
    ret, frame = cap.read()
    
    # Check if the frame was successfully read
    if not ret:
        break
        
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    # Display the processed frame
    cv2.imshow('Video', frame)
    
    # Wait for a key press to exit
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture object and close the window
cap.release()
cv2.destroyAllWindows()
