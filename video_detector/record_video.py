import cv2

# Set the video codec and frame rate
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 30

# Set the video resolution
width = 640
height = 480

# Create a VideoCapture object
cap = cv2.VideoCapture(0)  # Use 0 to capture video from the default camera

# Create a VideoWriter object
out = cv2.VideoWriter('output_video.avi', fourcc, fps, (width, height))

# Start the recording
start_time = cv2.getTickCount()  # Get the start time in ticks
while (cv2.getTickCount() - start_time) / cv2.getTickFrequency() < 10:  # Record for 10 seconds
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)

# Stop the recording
out.release()

# Release the VideoCapture object
cap.release()
