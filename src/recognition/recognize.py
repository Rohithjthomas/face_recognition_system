import os
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1

model = InceptionResnetV1(pretrained='vggface2').eval()
THRESHOLD = 1.0

def load_face(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (160, 160))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))
    return torch.tensor(img, dtype=torch.float32).unsqueeze(0)

def get_embedding(face_tensor):
    with torch.no_grad():
        emb = model(face_tensor).numpy()
    return emb[0]

def recognize_folder(test_folder, known_embeddings_path="data/embeddings/facebank.npy"):
    persons = np.load(known_embeddings_path, allow_pickle=True).item()

    for img_file in os.listdir(test_folder):
        img_path = os.path.join(test_folder, img_file)
        print(f"\n\n=== Testing: {img_file} ===")

        test_face = load_face(img_path)
        test_emb = get_embedding(test_face)

        best_person = None
        best_dist = 999

        for person_name, emb_list in persons.items():

            for emb in emb_list:
                dist = np.linalg.norm(test_emb - emb)

                if dist < best_dist:
                    best_dist = dist
                    best_person = person_name

        print(f"Closest person: {best_person}")
        print(f"Distance: {best_dist:.4f}")

        if best_dist < THRESHOLD:
            print(f" RESULT: {best_person.upper()}")
        else:
            print(" UNKNOWN PERSON")


if __name__ == "__main__":
    test_folder = "data/test"
    recognize_folder(test_folder)
