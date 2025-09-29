#!/usr/bin/env python3
"""
Debug preprocessing pipeline
"""
import cv2
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# Setup model and transform (same as in test.py)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    fixed_image_standardization
])

def debug_preprocessing(img_path, name):
    print(f"\n=== DEBUGGING {name} ===")

    # Load original image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Cannot load {img_path}")
        return None

    print(f"Original image shape: {img.shape}")
    print(f"Original pixel range: {img.min()}-{img.max()}")
    print(f"Original mean: {img.mean():.1f}")

    # Convert to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"RGB pixel range: {img_rgb.min()}-{img_rgb.max()}")
    print(f"RGB mean: {img_rgb.mean():.1f}")

    # Apply transforms step by step
    print("\n--- Step by step transform ---")

    # 1. ToPILImage
    from PIL import Image
    pil_img = Image.fromarray(img_rgb)
    print(f"1. PIL image size: {pil_img.size}")

    # 2. Resize
    resized = transforms.Resize((160, 160))(pil_img)
    print(f"2. After resize: {resized.size}")

    # 3. ToTensor
    tensor = transforms.ToTensor()(resized)
    print(f"3. After ToTensor - shape: {tensor.shape}")
    print(f"   Range: {tensor.min():.3f} to {tensor.max():.3f}")
    print(f"   Mean: {tensor.mean():.6f}, Std: {tensor.std():.6f}")
    print(f"   First few values: {tensor.flatten()[:5]}")

    # 4. fixed_image_standardization
    standardized = fixed_image_standardization(tensor)
    print(f"4. After standardization - shape: {standardized.shape}")
    print(f"   Range: {standardized.min():.3f} to {standardized.max():.3f}")
    print(f"   Mean: {standardized.mean():.6f}, Std: {standardized.std():.6f}")
    print(f"   First few values: {standardized.flatten()[:5]}")

    # 5. Add batch dimension and get embedding
    x = standardized.unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(x)

    embedding_np = embedding.cpu().numpy()[0]
    print(f"5. Final embedding:")
    print(f"   Shape: {embedding_np.shape}")
    print(f"   Norm: {np.linalg.norm(embedding_np):.6f}")
    print(f"   First 5: {embedding_np[:5]}")
    print(f"   Last 5: {embedding_np[-5:]}")

    return embedding_np

# Test with 2 different face crops
img1_path = "gallery/an/Screenshot 2025-09-27 165433.png"
img2_path = "gallery/nam/Screenshot 2025-09-27 170047.png"

emb1 = debug_preprocessing(img1_path, "AN")
emb2 = debug_preprocessing(img2_path, "NAM")

if emb1 is not None and emb2 is not None:
    print(f"\n=== COMPARISON ===")
    print(f"Are embeddings identical? {np.allclose(emb1, emb2)}")
    print(f"Max difference: {np.max(np.abs(emb1 - emb2)):.6f}")
    print(f"Cosine similarity: {np.dot(emb1/np.linalg.norm(emb1), emb2/np.linalg.norm(emb2)):.6f}")

