from data import extract_sample, testx, testy, display_sample
from model import load_protonet_conv
from tqdm import trange
from train import model
import matplotlib.pyplot as plt
import numpy as np

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

def test(model, test_x, test_y, n_way, n_support, n_query, test_episode):
    """
    Tests the protonet
    Args:
      model: trained model
      test_x (np.array): images of testing set
      test_y (np.array): labels of testing set
      n_way (int): number of classes in a classification task
      n_support (int): number of labeled examples per class in the support set
      n_query (int): number of labeled examples per class in the query set
      test_episode (int): number of episodes to test on
    """
    running_loss = 0.0
    running_acc = 0.0
    for episode in trange(test_episode):
        sample = extract_sample(n_way, n_support, n_query, test_x, test_y)
        loss, output = model.set_forward_loss(sample)
        running_loss += output['loss']
        running_acc += output['acc']
        if episode == 0:
            visualize_prediction(sample, output['y_hat'])
    avg_loss = running_loss / test_episode
    avg_acc = running_acc / test_episode
    print('Test results -- Loss: {:.4f} Acc: {:.4f}'.format(avg_loss, avg_acc))

n_way = 5
n_support = 5
n_query = 5

test_x = testx
test_y = testy

test_episode = 1000
test(model, test_x, test_y, n_way, n_support, n_query, test_episode)
