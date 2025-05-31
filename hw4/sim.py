import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader,random_split
from torchvision import transforms,datasets
from tqdm import tqdm
from sklearn.metrics import f1_score
from cifar import load_cifar10
import numpy as np
import os
from torchvision.models import resnet18
from PIL import Image
import matplotlib.pyplot as plt
import wandb
import yaml

 
transform_augment = transforms.Compose([
    transforms.RandomResizedCrop(size=32),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


class SimCLRDataset(Dataset):
    def __init__(self, x_data, transform):
        self.x_data = x_data
        self.transform = transform

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        image = self.x_data[idx].transpose(1, 2, 0)
        image = Image.fromarray((image * 255).astype(np.uint8))
        xi = self.transform(image)
        xj = self.transform(image)
        return xi, xj
    
def no_simclr_encoder(X_train, batch_size, epochs, lr, device):
    class Flatten(nn.Module):
        def forward(self, x):
            return x.view(x.size(0), -1)

    def conv_block(in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    encoder = nn.Sequential(
        conv_block(3, 64),
        conv_block(64, 64),
        conv_block(64, 64),
        conv_block(64, 64),
        Flatten()
    ).to(device)

    return encoder

class SimCLRNet(nn.Module):
    def __init__(self, encoder, device):
        super().__init__()
        self.encoder = encoder
        with torch.no_grad():
            dummy = torch.randn(1, 3, 32, 32).to(device)
            feat_dim = encoder(dummy).shape[1]
        dimens= self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        self.projector = nn.Sequential(
            nn.Linear(dimens, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        features = self.encoder(x)
        projections = self.projector(features)
        return F.normalize(projections, dim=1)
    
def train_simclr_encoder_resnet(simclr_model, X_train, batch_size=128, n_workers=4, epochs=10, lr=1e-3, device=None):

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = SimCLRDataset(X_train, transform_augment)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers)

    simclr_model = simclr_model.to(device)
    simclr_model.train()

    optimizer = torch.optim.Adam(simclr_model.parameters(), lr=lr)

    def nt_bxent_loss(zi, zj, temperature=0.1):
        batch_size = zi.size(0)
        zi = F.normalize(zi, dim=1)
        zj = F.normalize(zj, dim=1)

        sim_ij = torch.mm(zi, zj.t()) / temperature

        target = torch.arange(batch_size).to(zi.device)

        loss_i = F.cross_entropy(sim_ij, target)
        loss_j = F.cross_entropy(sim_ij.t(), target)

        loss = (loss_i + loss_j) / 2

        return loss

    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"SimCLR ResNet Pretrain Epoch {epoch+1}/{epochs}")
        for xi, xj in pbar:
            xi, xj = xi.to(device), xj.to(device)

            zi = simclr_model(xi) 
            zj = simclr_model(xj)
            loss = nt_bxent_loss(zi, zj, temperature=0.1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=total_loss / (pbar.n + 1))

    simclr_model.eval()

    return simclr_model.encoder
    
def train_simclr_encoder(X_train, batch_size=128, n_workers=4, epochs=30, lr=1e-3, device=None):

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = SimCLRDataset(X_train, transform_augment)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers)

    class Flatten(torch.nn.Module):
        def forward(self, x):
            return x.view(x.size(0), -1)

    def conv_block(in_c, out_c):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_c, out_c, 3, padding=1),
            torch.nn.BatchNorm2d(out_c),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )

    encoder = torch.nn.Sequential(
        conv_block(3, 64),
        conv_block(64, 64),
        conv_block(64, 64),
        conv_block(64, 64),
        Flatten()
    ).to(device)

    with torch.no_grad():
        dummy = torch.randn(1, 3, 32, 32).to(device)
        feat_dim = encoder(dummy).shape[1]

    projection_head = torch.nn.Sequential(
        torch.nn.Linear(feat_dim, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128)
    ).to(device)

    def nt_bxent_loss(x, pos_indices, temperature=0.1):
        assert len(x.size()) == 2
        diag_indices = torch.arange(x.size(0), device=x.device).unsqueeze(1).expand(-1, 2)
        pos_indices = torch.cat([pos_indices, diag_indices], dim=0)

        target = torch.zeros(x.size(0), x.size(0)).to(x.device)
        target[pos_indices[:, 0], pos_indices[:, 1]] = 1.0

        xcs = F.cosine_similarity(x[None, :, :], x[:, None, :], dim=-1)
        xcs[torch.eye(x.size(0), dtype=torch.bool, device=x.device)] = float("inf")

        loss = F.binary_cross_entropy((xcs / temperature).sigmoid(), target, reduction="none")

        target_pos = target.bool()
        target_neg = ~target_pos

        loss_pos = torch.zeros_like(loss).masked_scatter(target_pos, loss[target_pos])
        loss_neg = torch.zeros_like(loss).masked_scatter(target_neg, loss[target_neg])
        loss_pos = loss_pos.sum(dim=1)
        loss_neg = loss_neg.sum(dim=1)

        num_pos = target.sum(dim=1)
        num_neg = x.size(0) - num_pos

        return ((loss_pos / num_pos) + (loss_neg / num_neg)).mean()


    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(projection_head.parameters()), lr=lr)

    encoder.train()
    projection_head.train()

    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"SimCLR Pretrain Epoch {epoch+1}/{epochs}")
        for xi, xj in pbar:
            xi, xj = xi.to(device), xj.to(device)

            zi = projection_head(encoder(xi))
            zj = projection_head(encoder(xj))

            zi = F.normalize(zi, dim=1)
            zj = F.normalize(zj, dim=1)

            z = torch.cat([zi, zj], dim=0)
            N = zi.size(0)

            pos_indices = torch.stack([torch.arange(N, device=z.device),torch.arange(N, device=z.device) + N], dim=1)  # shape: [N, 2]

            loss = nt_bxent_loss(z, pos_indices, temperature=0.1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=total_loss / (pbar.n + 1))

    encoder.eval()
    return encoder

def init_scratch_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.encoder = resnet18(pretrained=True)
        in_features = self.encoder.fc.in_features
        self.encoder.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.encoder(x)

transformС = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class BaseTrainProcess:
    def __init__(self, hyp, X_train):
        self.best_loss = 1e100
        self.best_acc = 0.0
        self.current_epoch = -1
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.hyp = hyp
        self.X_train = X_train
        self.lr_scheduler = None
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.train_loader = None
        self.valid_loader = None
        self.train_losses = []
        self.val_losses=[]
        self.train_accs =[]
        self.val_accs = []
        self.init_params()


        full_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transformС)
        train_size = int(0.75 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        self.train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        self.valid_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    def _init_model(self):
        if self.hyp.get("use_resnet"):
            self.model = ResNet18Classifier(10)
        elif self.hyp.get('use_resnet_simclr'):
            base_encoder = resnet18(pretrained=False)
            simclr_model = SimCLRNet(base_encoder, self.device).to(self.device)
            simclr_model = train_simclr_encoder_resnet(simclr_model, self.X_train, batch_size=self.hyp['batch_size'], epochs=5, lr=self.hyp['lr'], device=self.device)
            encoder = simclr_model.encoder
            class SimCLRClassifier(nn.Module):
                def __init__(self, encoder, num_classes=10):
                    super().__init__()
                    self.encoder = encoder
                    for p in self.encoder.parameters():
                        p.requires_grad = False
                    self.encoder.fc = nn.Identity()
                    with torch.no_grad():
                        dummy = torch.randn(1, 3, 32, 32).to(self.encoder[0][0].weight.device)
                        feat_dim = encoder(dummy).shape[1]
                    self.classifier = nn.Linear(feat_dim, num_classes)

                def forward(self, x):
                    x = self.encoder(x)
                    return self.classifier(x)
            self.model = SimCLRClassifier(encoder).to(self.device)
        elif self.hyp.get('use_no'):
            self.encoder = no_simclr_encoder(self.X_train, batch_size=self.hyp['batch_size'],
                                                epochs=2, lr=self.hyp['lr'], device=self.device)
            class SimCLRClassifier(nn.Module):
                def __init__(self, encoder, num_classes=10):
                    super().__init__()
                    self.encoder = encoder
                    with torch.no_grad():
                        dummy = torch.randn(1, 3, 32, 32).to(self.encoder[0][0].weight.device)
                        feat_dim = encoder(dummy).shape[1]
                    self.classifier = nn.Linear(feat_dim, num_classes)

                def forward(self, x):
                    x = self.encoder(x)
                    return self.classifier(x)

            self.model = SimCLRClassifier(self.encoder, num_classes=10).to(self.device)
        else:
            self.encoder = train_simclr_encoder(self.X_train, batch_size=self.hyp['batch_size'],
                                                epochs=5, lr=self.hyp['lr'], device=self.device)

            class SimCLRClassifier(nn.Module):
                def __init__(self, encoder, num_classes=10):
                    super().__init__()
                    self.encoder = encoder
                    for p in self.encoder.parameters():
                        p.requires_grad = False
                    with torch.no_grad():
                        dummy = torch.randn(1, 3, 32, 32).to(self.encoder[0][0].weight.device)
                        feat_dim = encoder(dummy).shape[1]
                    self.classifier = nn.Linear(feat_dim, num_classes)

                def forward(self, x):
                    x = self.encoder(x)
                    return self.classifier(x)

            self.model = SimCLRClassifier(self.encoder, num_classes=10).to(self.device)
        if not self.hyp.get("use_resnet") or not self.hyp.get("use_resnet_simclr"):
            self.model.apply(init_scratch_weights)
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hyp['lr'],
                                          weight_decay=self.hyp['weight_decay'])
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min')

    def init_params(self):
        self._init_model()

    def train_step(self):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()

            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()

            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            total_loss += loss.item()
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        print(f"Epoch {self.current_epoch+1}: Train Acc={accuracy:.4f}, Train Loss={avg_loss:.4f}")
        return avg_loss, accuracy


    def valid_step(self):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in self.valid_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = self.criterion(logits, y)

                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.valid_loader)
        accuracy = correct / total
        print(f"Epoch {self.current_epoch+1}: Val Acc={accuracy:.4f}, Val Loss={avg_loss:.4f}")
        return avg_loss, accuracy
    
    def run(self):
        for epoch in range(self.hyp['epochs']):
            self.current_epoch = epoch
            train_loss, train_acc = self.train_step()
            val_loss, val_acc = self.valid_step()
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
        
        self.lr_scheduler.step(train_loss)

        torch.cuda.empty_cache()

def plot_metrics(train_processes, labels):
    plt.figure(figsize=(16, 6))
    for process, label in zip(train_processes, labels):
        plt.subplot(2, 2, 1)
        plt.plot(process.train_losses, label=f'{label} Train')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(process.train_accs, label=f'{label} Train')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(process.val_losses, label=f'{label} Val')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.plot(process.val_accs, label=f'{label} Val')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    params = yaml.safe_load(open("sim.yaml"))
    hyp = {
        'batch_size': params["batch_size"],
        'n_workers': params["n_workers"],
        'epochs': params["epochs"],
        'lr': params["lr"],
        'weight_decay': params["weight_decay"],
        'use_resnet': False,
        'use_no':False,
        'use_resnet_simclr':False
    }

    X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10("cifar_data")

    wandb.init(project="simclr-cifar10", config=hyp)

    print("Training SimCLR Encoder")
    trainer = BaseTrainProcess(hyp, X_train)
    trainer.run()

    print("Training No SimCLR Encoder")
    hyp['use_no'] = True
    trainer_no = BaseTrainProcess(hyp, X_train)
    trainer_no.run()

    print("Training ResNet18")
    hyp['use_resnet']=True
    trainer_resnet = BaseTrainProcess(hyp, X_train)
    trainer_resnet.run()

    print("Training ResNet18 FrozenEncoder")
    hyp['use_resnet_simclr'] = True
    trainer_resnetf = BaseTrainProcess(hyp, X_train)
    trainer_resnetf.run()

    plot_metrics(
        [trainer_no, trainer, trainer_resnet,trainer_resnetf],
        labels=['NoSimCLR', 'SimCLR', 'ResNet18','ResNet18SimCLR']
    )
