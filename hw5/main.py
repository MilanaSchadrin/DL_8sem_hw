import os
from PIL import Image
from collections import defaultdict
import torch.nn.functional as F
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics import roc_curve, auc
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import matplotlib.pyplot as plt
from glob import glob
import wandb
from sklearn.metrics import roc_curve
from datetime import datetime
from scipy.spatial.distance import cosine
from skimage.filters import threshold_otsu
from facenet_pytorch import MTCNN

from model import FaceNet
from prepare_data import filter_classes, TripletDataset
from train import*
from visualize_identify import*
from evaluate_model import*
from identify_face import*

wandb.init(project="hw5", entity="no312655-mipt")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset_root = os.path.join('lfw-funneled', 'lfw_funneled')
    valid_classes = filter_classes(dataset_root, min_samples=2)
    train_classes, test_classes = train_test_split(valid_classes, test_size=0.2, random_state=42)

    train_dataset = TripletDataset(dataset_root, train_classes, transform=train_transform)
    test_dataset = TripletDataset(dataset_root, test_classes, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    model = FaceNet().to(device)
    model.load_state_dict(torch.load('best_model_emb.pth'))
    model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    #train(model, train_loader, optimizer, device, epochs=20)
    #threshold, acc = evaluate(model, test_loader, device)

    #Пути для изображений
    ref_image = "basic/milana_basic.jpg"
    ref_person = 'milana'
    test_images = "dataset_2"

    matches, non_matches, threshold, ref_name = identify_face(ref_image, test_images, model, test_transform, device, ref_person)
    gt_file = "dataset_2/dataset_2/customer_dataset.txt"
    load_ground_truth("dataset_2/dataset_2/customer_dataset.txt")
    evaluate_predictions(gt_file, ref_person, matches, threshold, output_file='predictions_report.txt')

if __name__ == "__main__":
    main()