import torch
from facenet_pytorch import InceptionResnetV1
import numpy as np
import cv2
import os

print("SCRIPT STARTED")


# Load FaceNet 128D model
model = InceptionResnetV1(pretrained='vggface2').eval()

def load_face(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (160, 160))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    return img

def get_embedding(img_tensor):
    embedding = model(img_tensor).detach().numpy()
    return embedding[0]   # 128-D vector

def process_folder(folder):
    embeddings = {}
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        face = load_face(path)
        emb = get_embedding(face)
        embeddings[filename] = emb
        print("Embedded:", filename)
    return embeddings

if __name__ == "__main__":
    folder = "data/processed/rohith"
    embs = process_folder(folder)
    np.save("data/embeddings/rohith_128d.npy", embs)
    print("\nSaved embeddings to data/embeddings/rohith_128d.npy")
