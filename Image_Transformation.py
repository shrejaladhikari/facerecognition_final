import dlib
import cv2
import os

# Specify the path to the face detection model
predictor_model = "trainedmodel.dat"

# Path to the image file or folder
file_path = "../facerecognitionproj/trainingimages"

# Check if the model file exists
if not os.path.exists(predictor_model):
    print(
        f"Model file '{predictor_model}' not found. Please download it from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    exit(1)

# Initialize face detector and pose predictor
face_detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor(predictor_model)


# Function to align faces using Dlib
def align_face(image, face_rect):
    landmarks = face_pose_predictor(image, face_rect)
    left_eye = (landmarks.part(36).x, landmarks.part(36).y)
    right_eye = (landmarks.part(45).x, landmarks.part(45).y)

    # Align the face based on eye positions
    eyes_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
    angle = cv2.fastAtan2((right_eye[1] - left_eye[1]), (right_eye[0] - left_eye[0]))
    rotation_matrix = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
    aligned_face = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
    return aligned_face


# Process images
if os.path.isdir(file_path):  # Process all images in a folder
    for file_name in os.listdir(file_path):
        full_path = os.path.join(file_path, file_name)
        if os.path.isfile(full_path) and file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            print(f"Processing {file_name}...")
            image = cv2.imread(full_path)

            # Detect faces
            detected_faces = face_detector(image, 1)
            print(f"Found {len(detected_faces)} faces in {file_name}")

            for i, face_rect in enumerate(detected_faces):
                aligned_face = align_face(image, face_rect)
                output_path = os.path.join(file_path, f"aligned_face_{file_name}_{i}.jpg")
                cv2.imwrite(output_path, aligned_face)
                print(f"Aligned face saved to {output_path}")
else:
    print(f"Processing {file_path}...")
    image = cv2.imread(file_path)

    # Detect faces
    detected_faces = face_detector(image, 1)
    print(f"Found {len(detected_faces)} faces in {file_path}")

    for i, face_rect in enumerate(detected_faces):
        aligned_face = align_face(image, face_rect)
        output_path = f"aligned_face_{i}.jpg"
        cv2.imwrite(output_path, aligned_face)
        print(f"Aligned face saved to {output_path}")
