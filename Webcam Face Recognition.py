import numpy as np
import face_recognition
import os
import cv2
import csv
from datetime import datetime

path = '../facerecognitionproj/trainingimages'
images = []
classNames = []

# Load images and handle errors
myList = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    if curImg is not None:
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
    else:
        print(f"Failed to load image: {cl}")

# Encoding function
def findEncodings(images):
    encodeList = []
    for img in images:
        if img is not None:
            try:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                encodes = face_recognition.face_encodings(img)
                if len(encodes) > 0:
                    encodeList.append(encodes[0])
                else:
                    print("No face detected in the image.")
            except Exception as e:
                print(f"Error processing image: {e}")
        else:
            print("Skipping empty image.")
    return encodeList

encodeListKnown = findEncodings(images)
print('Encoding Done')

# Initialize attendance file
attendance_file = "attendance.csv"
if not os.path.exists(attendance_file):
    with open(attendance_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Date", "Time"])

# Mark attendance
def markAttendance(name):
    with open(attendance_file, "r+", newline="") as file:
        existing_data = file.readlines()
        today_date = datetime.now().strftime("%Y-%m-%d")
        time_now = datetime.now().strftime("%H:%M:%S")

        # Check if name and today's date are already recorded
        attendance_logged = any(name in line and today_date in line for line in existing_data)

        if not attendance_logged:
            writer = csv.writer(file)
            writer.writerow([name, today_date, time_now])
            print(f"Attendance marked for {name} at {time_now}")

video_capture = cv2.VideoCapture(0)

while True:
    success, img = video_capture.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS)

    for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDist = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDist)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            markAttendance(name)

            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2 + 60, y2), (0, 0, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Video', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
