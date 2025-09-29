#!/usr/bin/env python3
"""
Quick debug script to test embeddings from different images
"""
import cv2
import numpy as np
from test import face_embedding, detect_and_crop_face, l2_norm, cosine

def test_embeddings():
    # Test với 2 ảnh khác nhau
    img1_path = "gallery/an/Screenshot 2025-09-27 162032.png"
    img2_path = "gallery/nam/Screenshot 2025-09-27 170006.png"

    print("=== TESTING EMBEDDINGS ===")

    # Load và process ảnh 1
    print(f"\n1. Processing {img1_path}")
    img1 = cv2.imread(img1_path)
    if img1 is not None:
        print(f"   Original shape: {img1.shape}")
        print(f"   Mean pixel: {img1.mean():.1f}")

        face1 = detect_and_crop_face(img1, debug=True)
        if face1 is not None:
            emb1 = face_embedding(face1, debug=True)
            if emb1 is not None:
                print(f"   Embedding shape: {emb1.shape}")
                print(f"   Embedding norm: {np.linalg.norm(emb1):.6f}")
                print(f"   First 5 values: {emb1[:5]}")

    # Load và process ảnh 2
    print(f"\n2. Processing {img2_path}")
    img2 = cv2.imread(img2_path)
    if img2 is not None:
        print(f"   Original shape: {img2.shape}")
        print(f"   Mean pixel: {img2.mean():.1f}")

        face2 = detect_and_crop_face(img2, debug=True)
        if face2 is not None:
            emb2 = face_embedding(face2, debug=True)
            if emb2 is not None:
                print(f"   Embedding shape: {emb2.shape}")
                print(f"   Embedding norm: {np.linalg.norm(emb2):.6f}")
                print(f"   First 5 values: {emb2[:5]}")

    # So sánh
    if 'emb1' in locals() and 'emb2' in locals() and emb1 is not None and emb2 is not None:
        print(f"\n3. Comparison:")
        print(f"   Are identical? {np.allclose(emb1, emb2)}")
        print(f"   Max difference: {np.max(np.abs(emb1 - emb2)):.6f}")
        similarity = cosine(emb1, emb2, debug=True)
        print(f"   Cosine similarity: {similarity:.6f}")
    else:
        print("\n3. ERROR: Could not generate both embeddings")

if __name__ == "__main__":
    test_embeddings()
