import os
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1

# Load FaceNet 128D model
model = InceptionResnetV1(pretrained='vggface2').eval()

THRESHOLD = 1.0  # Euclidean threshold

def load_face(path):
    img = cv2.imread(path)
    if img is None:
        print(f"❌ Cannot load image: {path}")
        return None

    img = cv2.resize(img, (160, 160))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    return img

def get_embedding(img_tensor):
    with torch.no_grad():
        emb = model(img_tensor).numpy()
    return emb[0]

def recognize_single(test_image_path, known_dict):
    face_tensor = load_face(test_image_path)
    if face_tensor is None:
        return

    test_emb = get_embedding(face_tensor)

    best_match = None
    best_dist = 999

    for filename, emb in known_dict.items():
        dist = np.linalg.norm(emb - test_emb)
        print(f"{filename} → distance = {dist:.4f}")

        if dist < best_dist:
            best_dist = dist
            best_match = filename

    print("\n🟩 BEST MATCH:", best_match)
    print("📏 DISTANCE:", best_dist)

    if best_dist < THRESHOLD:
        print("🚀 RESULT: SAME PERSON\n")
    else:
        print("❌ RESULT: UNKNOWN PERSON\n")

def recognize(test_path, known_embeddings_path):
    # Load known embeddings
    known_dict = np.load(known_embeddings_path, allow_pickle=True).item()

    # If user gave a folder
    if os.path.isdir(test_path):
        print(f"📁 Scanning folder: {test_path}\n")
        for file in os.listdir(test_path):
            full_path = os.path.join(test_path, file)
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                print(f"\n=== Testing: {file} ===")
                recognize_single(full_path, known_dict)
    else:
        recognize_single(test_path, known_dict)

if __name__ == "__main__":
    test_path = "data/test"  # FOLDER
    known_embs = "data/embeddings/rohith_128d.npy"

    recognize(test_path, known_embs)
