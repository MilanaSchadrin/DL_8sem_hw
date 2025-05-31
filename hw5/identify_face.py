import os
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from visualize_identify import *
import numpy as np

def compute_embeddings(model, image_paths, transform, device, batch_size=32):
    model.eval()
    embeddings = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []
        for path in batch_paths:
            try:
                img = Image.open(path).convert('RGB')
                img = transform(img)
                batch_images.append(img)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                continue
        if not batch_images:
            continue
        batch_tensor = torch.stack(batch_images).to(device)
        with torch.no_grad():
            batch_emb = model(batch_tensor).cpu().numpy()
            batch_emb = batch_emb / np.linalg.norm(batch_emb, axis=1, keepdims=True)
            embeddings.extend(batch_emb)
    return np.array(embeddings)

def identify_face(reference_path, test_folder, model, transform, device, ref_im, threshold=None):
    test_paths = sorted(get_image_paths(test_folder))[:15]
    if len(test_paths) != 15:
        raise ValueError(f"Задано 15 изображений, найдено {len(test_paths)}")
    ref_name = os.path.basename(reference_path)
    #print(f"Эталонное изображение: {ref_name}")
    #print("Тестовые изображения:", [os.path.basename(p) for p in test_paths])
    ref_emb = compute_embeddings(model, [reference_path], transform, device)[0]
    test_embs = compute_embeddings(model, test_paths, transform, device)
    ref_emb = ref_emb / np.linalg.norm(ref_emb)
    test_embs = test_embs / np.linalg.norm(test_embs, axis=1, keepdims=True)
    distances = 1 - np.dot(test_embs, ref_emb)
    distances_dict = dict(zip([os.path.basename(p) for p in test_paths], distances))
    #адаптивный расчет порога (если порог не задан)
    if threshold is None:
        sorted_distances = np.sort(distances)
        top3_max = sorted_distances[3]
        threshold = min(top3_max * 1.2, 0.65)
        #print(f"\nАвтоматический порог: {threshold:.4f}")
        #print(f"(На основе 3x ближайших изображений: {sorted_distances[:3]})")
    matches = []
    non_matches = []
    for path, dist in zip(test_paths, distances):
        if dist <= threshold:
            matches.append((path, dist))
        else:
            non_matches.append((path, dist))
    visualize_comparison(reference_path, matches, non_matches, threshold,  ref_im)
    return matches, non_matches, threshold, ref_name
