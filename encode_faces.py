import cv2
import face_recognition
import os
import numpy as np
import pickle

path = 'student_images'
images = []
classNames = []

if not os.path.exists(path):
    print(f"Error: {path} folder missing! Create it and add photos.")
    exit(1)

mylist = os.listdir(path)
if not mylist:
    print(f"Error: No images in {path}! Add at least one photo named like 'john_doe.jpg'.")
    exit(1)

for cl in mylist:
    if not cl.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    curImg = cv2.imread(f'{path}/{cl}')
    if curImg is None:
        print(f"Warning: Skipping invalid image {cl}")
        continue
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])  # Name from filename
    print(f"Loaded: {cl} -> {classNames[-1]}")

def findEncodings(images):
    encodeList = []
    for img in images:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            encoded_face = face_recognition.face_encodings(img_rgb)[0]  # First face
            encodeList.append(encoded_face)
            print(f"Encoded: {len(encodeList)} faces so far")
        except IndexError:
            print(f"Warning: No face found in image—skipping.")
    return encodeList

encoded_face_train = findEncodings(images)
if not encoded_face_train:
    print("Error: No valid faces encoded! Check images for clear frontal faces.")
    exit(1)

# Save both lists
with open('encodings.pkl', 'wb') as f:
    pickle.dump(encoded_face_train, f)
    pickle.dump(classNames, f)
print(f"Encodings saved! {len(encoded_face_train)} faces for {len(classNames)} people.")