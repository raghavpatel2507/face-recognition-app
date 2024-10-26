from cv2 import imread,resize,


def load_face_encoding(image_path, resize_factor=0.25):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return None
    img = cv2.resize(img, (0, 0), fx=resize_factor, fy=resize_factor)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb_img)
    return encodings[0] if encodings else None


def format_face_encoding(encoding):
    return list(map(float, encoding))


register_image_path=""
face_encoding = load_face_encoding(register_image_path, resize_factor=0.5)
if face_encoding is not None:
        print(format_face_encoding(face_encoding))