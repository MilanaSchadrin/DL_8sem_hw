from tqdm.auto import tqdm
import torch
from stuff import convert_batch
import math
import wandb
import os

def do_epoch(model, criterion, data_iter, epoch, optimizer=None, name=None):
    
    os.makedirs("models", exist_ok=True)
    is_train = optimizer is not None
    model.train(is_train)
    name = name or ""
    total_loss = 0
    total_batches = len(data_iter)

    with torch.autograd.set_grad_enabled(is_train), tqdm(total=total_batches) as pbar:
        for step, batch in enumerate(data_iter):
            src_input, tgt_input, src_mask, tgt_mask = convert_batch(batch)

            logits = model(src_input, tgt_input[:, :-1], src_mask, tgt_mask[:, :-1, :-1])
            logits = logits.contiguous().view(-1, logits.shape[-1])
            targets = tgt_input[:, 1:].contiguous().view(-1)
            loss = criterion(logits, targets)

            total_loss += loss.item()

            if is_train:
                optimizer.optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                wandb.log({"loss": loss.item()}, step=step + epoch * total_batches)

            pbar.update()
            pbar.set_description(f'{name:<8} Loss: {loss.item():.5f} | PPX: {math.exp(loss.item()):.2f}')

            if step % 100 == 0 or total_loss < 0.1 * step:
                torch.save(model.state_dict(), "models/model_temp.pt")
                if total_loss < 0.1 * step:
                    break

        avg_loss = total_loss / total_batches
        pbar.set_description(f'{name:<8} AvgLoss: {avg_loss:.5f} | AvgPPX: {math.exp(avg_loss):.2f}')
        pbar.refresh()

    torch.save(model.state_dict(), f"models/model_epoch{epoch}.pt")
    return avg_loss


def fit(model, criterion, optimizer, train_iter, start_epoch=0, epochs_count=30, val_iter=None):
    wandb.init(
        config=dict(epochs=epochs_count, label_smoothing=0.1),
        project="Death is here",
        name="AAAA"
    )

    best_val_loss = float('inf')
    step = 0

    for epoch in range(start_epoch, start_epoch + epochs_count):
        epoch_label = f"[{epoch + 1} / {start_epoch + epochs_count}]"
        train_loss = do_epoch(model, criterion, train_iter, epoch, optimizer, name=epoch_label + " Train:")

        metrics = {"train_loss": train_loss}
        step += len(train_iter)

        if val_iter is not None:
            val_loss = do_epoch(model, criterion, val_iter, epoch, optimizer=None, name=epoch_label + " Val:")
            metrics["val_loss"] = val_loss
            best_val_loss = min(best_val_loss, val_loss)
            step += len(val_iter)

        wandb.log(metrics, step=step)

    os.makedirs("results", exist_ok=True)
    with open("results/wandb_run_id.txt", "w") as f:
        print(wandb.run.id, file=f)

    with open("results/val_loss_best.txt", "w") as f:
        print(best_val_loss, file=f)