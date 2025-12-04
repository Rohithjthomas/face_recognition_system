import os
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1

model = InceptionResnetV1(pretrained='vggface2').eval()

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
    return emb[0]    # 128D

def build_all_embeddings(processed_root="data/processed"):
    persons = {}

    for person_name in os.listdir(processed_root):
        person_folder = os.path.join(processed_root, person_name)
        if not os.path.isdir(person_folder):
            continue

        print(f"\n Processing person: {person_name}")
        embeddings = []

        for img_file in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_file)

            face = load_face(img_path)
            emb = get_embedding(face)

            embeddings.append(emb)
            print(f"   Embedded → {img_file}")

        persons[person_name] = embeddings

    # Save dictionary
    os.makedirs("data/embeddings", exist_ok=True)
    np.save("data/embeddings/facebank.npy", persons)


    print("\n Saved embeddings → data/embeddings/facebank.npy")


if __name__ == "__main__":
    build_all_embeddings()

