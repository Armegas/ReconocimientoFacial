import cv2
import os
import numpy as np

def train_model(uploads_dir):
    """
    Trains an LBPH Face Recognizer using images in the uploads directory.
    Returns:
        recognizer: Trained cv2.face.LBPHFaceRecognizer_create() object (or None if no faces)
        names_dict: Dictionary mapping internal ID (int) to Name (str)
        display_ids_dict: Dictionary mapping internal ID (int) to Display ID (str)
    """
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    faces = []
    ids = []
    names_dict = {}       # map internal_id -> Name
    display_ids_dict = {} # map internal_id -> User provided ID
    
    # Internal mapping management
    # We need a stable mapping from (Name, UserID) -> internal_int_id
    # For simplicity in this demo, we'll regenerate it each reload.
    persons_map = {} # key: (name, userid) -> value: internal_id
    current_internal_id = 0

    if not os.path.exists(uploads_dir):
        return None, {}, {}

    count_images = 0
    for filename in os.listdir(uploads_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(uploads_dir, filename)
            
            # Parse filename Name_ID.ext
            name_part = os.path.splitext(filename)[0]
            parts = name_part.split('_')
            if len(parts) >= 2:
                name = parts[0]
                user_id = parts[1]
            else:
                name = name_part
                user_id = "N/A"
            
            # Create unique key for this person
            person_key = (name, user_id)
            if person_key not in persons_map:
                persons_map[person_key] = current_internal_id
                names_dict[current_internal_id] = name
                display_ids_dict[current_internal_id] = user_id
                current_internal_id += 1
            
            internal_id = persons_map[person_key]

            try:
                # Read image in grayscale
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                # Detect faces
                faces_detected = detector.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)

                for (x, y, w, h) in faces_detected:
                    faces.append(img[y:y+h, x:x+w])
                    ids.append(internal_id)
                    count_images += 1
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    if count_images > 0:
        recognizer.train(faces, np.array(ids))
        return recognizer, names_dict, display_ids_dict
    else:
        return None, {}, {}
