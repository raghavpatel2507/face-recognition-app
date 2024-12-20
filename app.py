print('start the code')
from datetime import datetime
print('import modeule',datetime.now())
from cv2 import imread,resize,cvtColor,COLOR_BGR2RGB
from face_recognition import face_encodings,face_locations,face_distance
import numpy as np
import json
import os
import sys

def load_face_encoding(image_path, resize_factor=0.25):
    print('load the image',datetime.now())
    img = imread(image_path)
    print('read the image',datetime.now())
    if img is None:
        print(f"Failed to load image: {image_path}")
        return None
    print('resize image',datetime.now())
    img = resize(img, (0, 0), fx=resize_factor, fy=resize_factor)
    print('resize image successfully',datetime.now())
    rgb_img = cvtColor(img, COLOR_BGR2RGB)
    print('change the colour',datetime.now())
    print('load the face encodings',datetime.now())
    encodings = face_encodings(rgb_img)
    print('end the face encodings',datetime.now())
    return encodings[0] if encodings else None

def format_face_encoding(encoding):
    return list(map(float, encoding))

def parse_face_encodings(employee_json):
    try:
        data = json.loads(employee_json)
        encodings = {}
        for emp in data["lstEmpBiometric"]:
            empid = emp['EmployeeID']
            contract_id = emp['ContractorGroupStaffID']
            face_float_data = json.loads(emp['FaceFloatData'][0])
            encodings[empid] = (contract_id, [np.array(encoding, dtype=np.float64) for encoding in face_float_data])
        return encodings
    except (json.JSONDecodeError, SyntaxError, ValueError) as e:
        raise ValueError(f"Error parsing face encodings: {e}")

def recognize_faces(image, known_face_encodings):
    face_locations = face_locations(image)
    face_encodings = face_encodings(image, face_locations)
    recognized_info = []
    
    for face_encoding in face_encodings:
        found = False
        for empid, (contract_id, encodings) in known_face_encodings.items():
            distances = face_distance(encodings, face_encoding)
            if np.any(distances < 0.4):
                recognized_info.append((empid, contract_id))
                found = True
                break
        if not found:
            recognized_info.append((None, None))

    return recognized_info

def main(image_path, employee_file):
    try:
        if not os.path.exists(employee_file):
            raise Exception(f"JSON file not found: {employee_file}")
        
        with open(employee_file, 'r') as f:
            employee_json = f.read()

        known_face_encodings = parse_face_encodings(employee_json)

        target_image = imread(image_path)
        if target_image is None:
            raise Exception(f"Failed to load target image: {image_path}")
        
        target_image_resized = resize(target_image, (0, 0), fx=0.5, fy=0.5)
        target_image_rgb = cvtColor(target_image_resized, COLOR_BGR2RGB)

        recognized_info = recognize_faces(target_image_rgb, known_face_encodings)

        for empid, contract_id in recognized_info:
            if empid or contract_id:
                print(f"{empid},{contract_id}")
            else:
                print("No matching employee found")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    register_image_path = "mh1.jpg"
    action = "R"

    if action == 'R':
        print('load the function',datetime.now())
        face_encoding = load_face_encoding(register_image_path, resize_factor=0.5)
        print('end load the face encodings function',datetime.now())
        if face_encoding is not None:
            print(format_face_encoding(face_encoding))
    else:
        employee_file = sys.argv[3]
        main(register_image_path, employee_file)