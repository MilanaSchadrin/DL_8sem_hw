import wandb
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

def train(model, dataloader, optimizer, device, epochs=10, margin=0.5):
    wandb.init(project="hw5", entity="no312655-mipt")
    wandb.config.update({
        "epochs": epochs,
        "margin": margin,
        "optimizer": type(optimizer).__name__,
        "learning_rate": optimizer.param_groups[0]['lr']
    })
    wandb.watch(model, log="all", log_freq=10)
    history = {
        'train_loss': [],
        'train_acc': [],
        'lr': []
    }

    model.train()
    criterion = nn.TripletMarginLoss(margin=margin, p=2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        batch_losses = []
        batch_accs = []
        pbar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{epochs}', leave=True)
        for batch_idx, (anchor, positive, negative) in enumerate(pbar):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            optimizer.zero_grad()
            anchor_out = model(anchor)
            positive_out = model(positive)
            negative_out = model(negative)
            loss = criterion(anchor_out, positive_out, negative_out)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            with torch.no_grad():
                d_ap = F.pairwise_distance(anchor_out, positive_out)
                d_an = F.pairwise_distance(anchor_out, negative_out)
                correct += (d_ap < d_an).sum().item()
                total += anchor.size(0)
                batch_loss = loss.item()
                batch_acc = (d_ap < d_an).float().mean().item()
                batch_losses.append(batch_loss)
                batch_accs.append(batch_acc)

            if batch_idx % 10 == 0:  # Логируем каждые 10 батчей
                wandb.log({
                    "batch_loss": loss.item(),
                    "batch_accuracy": (d_ap < d_an).float().mean().item(),
                    "learning_rate": optimizer.param_groups[0]['lr'], "epoch": epoch + (batch_idx/len(dataloader))
                })

            avg_loss = total_loss / (len(pbar) + 1e-8)
            acc = correct / (total + 1e-8) * 100
            pbar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'Accuracy': f'{acc:.2f}%',
                'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
        avg_loss = total_loss / len(dataloader)
        acc = correct / total * 100

        history['train_loss'].append(avg_loss)
        history['train_acc'].append(acc / 100)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        wandb.log({
            "epoch": epoch,
            "epoch_loss": avg_loss,
            "epoch_accuracy": acc / 100,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        plot_training_metrics(history, epoch)
        scheduler.step(avg_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'best_model_emb.pth')
        print(f"Epoch {epoch + 1} complete. Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%")
    wandb.finish()

def plot_training_metrics(history, current_epoch):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(history['lr'], label='Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'training_metrics_epoch_{current_epoch}.png')
    wandb.log({"training_metrics": wandb.Image(f'training_metrics_epoch_{current_epoch}.png')})
    plt.close()
