import cv2

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the video
cap = cv2.VideoCapture('output_video.avi')

# Create a list to store the detected images
images = []

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

    # For each detected face, save the corresponding image
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        # Resize the face to a fixed size (100 x 100)
        face = cv2.resize(face, (100, 100))
        images.append(face)

# Create a grid of images
num_images = len(images)
num_cols = 5
num_rows = (num_images + num_cols - 1) // num_cols  # Round up to the nearest integer
grid = None
for i, image in enumerate(images):
    if i % num_cols == 0:
        if grid is not None:
            cv2.imshow('Grid', grid)
            cv2.waitKey(0)
        grid = image
    else:
        grid = cv2.hconcat([grid, image])

# Add empty cells to the last row to fill the grid
num_empty_cells = num_cols - (num_images % num_cols)
for i in range(num_empty_cells):
    empty_cell = 255 * np.ones((100, 100, 3), dtype=np.uint8)
    grid = cv2.hconcat([grid, empty_cell])

# Add empty rows to the bottom of the grid to make it rectangular
num_empty_rows = num_rows * num_cols - num_images
for i in range(num_empty_rows):
    empty_row = 255 * np.ones((100 * num_cols, 100, 3), dtype=np.uint8)
    grid = cv2.vconcat([grid, empty_row])

# Display the grid
cv2.imshow('Grid', grid)
cv2.waitKey(0)

# Release the video capture object and close the window
cap.release()
cv2.destroyAllWindows()
