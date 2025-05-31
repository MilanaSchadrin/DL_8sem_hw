from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import wandb 
from data import extract_sample, trainx, trainy
from model import load_protonet_conv
from tqdm import trange
import torch
import yaml

params = yaml.safe_load(open("params.yaml"))
wandb.init(project="protonet-omniglot", config=params) 

model = load_protonet_conv(
    x_dim=(3, 28, 28),
    hid_dim=64,
    z_dim=64
)
optimizer = optim.Adam(model.parameters(), lr=0.001)
wandb.watch(model, log="gradients", log_freq=100)
def train(model, optimizer, train_x, train_y, n_way, n_support, n_query, max_epoch, epoch_size):
    """
    Trains the protonet
    Args:
      model
      optimizer
      train_x (np.array): images of training set
      train_y(np.array): labels of training set
      n_way (int): number of classes in a classification task
      n_support (int): number of labeled examples per class in the support set
      n_query (int): number of labeled examples per class in the query set
      max_epoch (int): max epochs to train on
      epoch_size (int): episodes per epoch
    """
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5, last_epoch=-1)
    epoch = 0 
    stop = False 
    while epoch < max_epoch and not stop:
        running_loss = 0.0
        running_acc = 0.0

        for episode in trange(epoch_size, desc="Epoch {:d} train".format(epoch + 1)):
            sample = extract_sample(n_way, n_support, n_query, train_x, train_y)
            optimizer.zero_grad()
            loss, output = model.set_forward_loss(sample)
            running_loss += output['loss']
            running_acc += output['acc']

            loss.backward()
            optimizer.step()
        
        epoch_loss = running_loss / epoch_size
        epoch_acc = running_acc / epoch_size
        print('Epoch {:d} -- Loss: {:.4f} Acc: {:.4f}'.format(epoch+1,epoch_loss, epoch_acc))
        wandb.log({"epoch_loss": epoch_loss,
                   "epoch_acc" : epoch_acc,
                   "epoch"     : epoch}, step=(epoch+1)*epoch_size)
        epoch += 1
        scheduler.step()
    torch.save(model.state_dict(), 'protonet_model.pt')



if __name__ == "__main__":
    n_way = params["n_way"]
    n_support =params["n_support"]
    n_query = params["n_query"]

    train_x = trainx
    train_y = trainy

    max_epoch = params["max_epoch"]
    epoch_size = params["epoch_size"]

    train(model, optimizer, train_x, train_y, n_way, n_support, n_query, max_epoch, epoch_size)
