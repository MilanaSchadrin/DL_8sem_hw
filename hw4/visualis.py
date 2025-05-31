import os
import random
import string
import matplotlib.pyplot as plt
from data import extract_sample
import numpy as np

def random_filename(length=10):
    return ''.join(random.choices(string.digits, k=length)) + '.png'

def visualize_prediction(sample, y_hat, save_path='predictions.png'):
    """
    Визуализация предсказаний модели на одном эпизоде
    Args:
        sample (dict): содержит 'images' — тензор формы [n_way, n_support + n_query, 3, 28, 28]
        y_hat (Tensor): предсказания модели (форма: [n_way, n_query])
    """
    n_way = sample['n_way']
    n_query = sample['n_query']
    query_images = sample['images'][:, sample['n_support']:].cpu().numpy()

    fig, axs = plt.subplots(n_way, n_query, figsize=(n_query * 2, n_way * 2))

    for i in range(n_way):
        for j in range(n_query):
            ax = axs[i, j]
            img = query_images[i, j].transpose(1, 2, 0).astype(np.uint8)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f'True: {i}\nPred: {y_hat[i, j].item()}')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def visualize_n_predictions(model, test_x, test_y, n_way, n_support, n_query, n_images=30, save_dir='test_vis'):

    os.makedirs(save_dir, exist_ok=True)

    saved_count = 0
    while saved_count < n_images:
        sample = extract_sample(n_way, n_support, n_query, test_x, test_y)
        _, output = model.set_forward_loss(sample)

        query_images = sample['images'][:, n_support:].cpu().numpy()
        y_hat = output['y_hat']

        for i in range(n_way):
            for j in range(n_query):
                if saved_count >= n_images:
                    break
                img = query_images[i, j].transpose(1, 2, 0).astype(np.uint8)

                fig, ax = plt.subplots()
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(f'True: {i}, Pred: {y_hat[i, j].item()}', fontsize=8)

                filename = random_filename()
                save_path = os.path.join(save_dir, filename)
                plt.savefig(save_path)
                plt.close(fig)

                saved_count += 1

    print(f"Saved {n_images} prediction images to '{save_dir}/'")