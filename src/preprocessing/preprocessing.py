import cv2
import os
import mediapipe as mp

mp_face = mp.solutions.face_detection

def load_image(path):
    img = cv2.imread(path)
    return img

def detect_face(image):
    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.6) as detector:
        results = detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.detections:
            return None
        
        # Take first face only
        det = results.detections[0]
        bbox = det.location_data.relative_bounding_box

        h, w, _ = image.shape

        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        w_box = int(bbox.width * w)
        h_box = int(bbox.height * h)

        return (x, y, w_box, h_box)

def crop_and_resize(image, box, size=(160, 160)):
    x, y, w, h = box
    face = image[y:y+h,x:x+w]
    face = cv2.resize(face,size)
    return face

def preprocess_image(input_path, output_path):
    img = load_image(input_path)
    box = detect_face(img)

    if box is None:
        print(f"Skipping {input_path} - no face or multiple faces")
        return False

    face = crop_and_resize(img, box)

    cv2.imwrite(output_path, face)
    return True


def preprocess_all_people(raw_root="data/raw", processed_root="data/processed"):

    for person_name in os.listdir(raw_root):
        raw_folder = os.path.join(raw_root, person_name)
        processed_folder = os.path.join(processed_root, person_name)

        # skip files, accept only folders
        if not os.path.isdir(raw_folder):
            continue

        print(f"\n Processing person: {person_name}")

        if not os.path.exists(processed_folder):
            os.makedirs(processed_folder)

        # Run preprocessing for each person
        for filename in os.listdir(raw_folder):
            input_path = os.path.join(raw_folder, filename)
            output_path = os.path.join(processed_folder, filename)

            success = preprocess_image(input_path, output_path)
            if success:
                print(f" Processed {filename}")
            else:
                print(f" Failed {filename}")


if __name__ == "__main__":
    print("\n Starting preprocessing for ALL persons...")
    preprocess_all_people()
    print("\n DONE — All faces processed.")
