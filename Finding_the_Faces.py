import sys
import os
import dlib
from skimage import io
import cv2

# Check for command-line arguments or use a default
if len(sys.argv) < 2:
    folder_path = "../facerecognitionproj/trainingimages"
    image_files = os.listdir(folder_path)
    if not image_files:
        print(f"No images found in folder: {folder_path}")
        sys.exit(1)
    file_name = os.path.join(folder_path, image_files[0])
    print(f"No image specified. Using the first image in {folder_path}: {file_name}")
else:
    file_name = sys.argv[1]

# Create a HOG face detector using the built-in dlib class
face_detector = dlib.get_frontal_face_detector()

# Load the image into an array
image = io.imread(file_name)

# Run the HOG face detector on the image data
detected_faces = face_detector(image, 1)

print("I found {} faces in the file {}".format(len(detected_faces), file_name))

# Convert the image for OpenCV compatibility
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# Loop through each face we found in the image
for i, face_rect in enumerate(detected_faces):
    print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(
        i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))

    # Draw a box around each face we found
    x1, y1, x2, y2 = face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Display the image with detections
cv2.imshow('Face Detection', image)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
