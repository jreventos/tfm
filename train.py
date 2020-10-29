
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision import transforms
from torch import optim
import torch
import argparse
import time
import os
from tfm.model import  metrics
from tfm.model.Unet import *
from tfm.dataset import*
from tfm.utils.patches import *
import matplotlib.pyplot as plt

from tfm.preprocessing import custom_normalize


from monai.losses import DiceLoss, GeneralizedDiceLoss
# Parser
parser = argparse.ArgumentParser()
# Set seed
#parser.add_argument("--seed", type=int, default=42, help="Seed")

# Data loader parameters
parser.add_argument("--path", type=str, default='/Users/jreventos/Desktop/TFM/tfm/patients_data2', help="Path to the training and val data")
parser.add_argument("--mean", type=list, default=21.201036, help="Mean of the data set MRI volumes")
parser.add_argument("--sd", type=list, default= 40.294086, help="Standard deviation of the data set MRI volumes")
parser.add_argument("--patch_dim", type=list, default=[60,60,32], help="Patches dimensions")
parser.add_argument("--stepsize", type=int, default=64, help="patches overlap stepsize")



# Model parameters
parser.add_argument("--is_load", type=bool, default= False, help="weights initialization")
parser.add_argument("--load_path", type=str, default= '', help="path to the weights net initialization")
parser.add_argument("--lr", type=int, default= 0.0001, help="learning rate")



# In case of transfer models
#parser.add_argument("--feature_extract", type=bool, default=False, help="If true do features extraction, otherwise finetuning")
#parser.add_argument("--fully_connected", type=str, default=1, help="Type of fully connected after convolution")
#parser.add_argument("--fine_tuning_prop", type=str, default='complete', help='Proportion of the net unfreezed')

# Training parameters
parser.add_argument("--epoch", type=int, default=1, help="Number of epochs")
parser.add_argument("--epoch_init", type=int, default=0, help="Number of epochs where we want to initialize")
parser.add_argument("--batch", type=int, default=1, help="Number of examples in batch")
parser.add_argument("--verbose", type=int, default=1, help="Verbose, when we want to do a print")

parser.add_argument("--early_stopping", type=int, default=5, help="Number of epochs without improvement to stop")
parser.add_argument("--save_models", type=bool, default=False, help='If true models will be saved')

opt = parser.parse_args()

# Define model name
model_name = '_Unet'.join(['epoch_',str(opt.epoch),'batch_', str(opt.batch)])

class average_metrics(object):
    """
    Average metrics class, use the update to add the current metric and self.avg to get the avg of the metric
    """
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_epoch(data_loader, model, criterion, optimizer=None, mode_train=True):

    total_batch = len(data_loader)
    batch_loss = average_metrics()

    if mode_train:
        model.train()
    else:
        model.eval()

    for batch_idx, (data, y) in enumerate(data_loader):
        data = data.to(device)
        y = y.to(device)

        # Reset gradients
        if mode_train:
            optimizer.zero_grad()

        # Get prediction

        out = model(data)
        #print(out)
        #print(out.shape)

        # Get loss
        loss = criterion(out, y)

        # Backpropagate error
        loss.backward()

        # Update loss
        batch_loss.update(loss.item())

        # Optimize
        if mode_train:
            optimizer.step()

        # Log
        if (batch_idx + 1) % opt.verbose == 0 and mode_train:
            print(f'Iteration {(batch_idx + 1)}/{total_batch} - Loss: {batch_loss.val} ')

        if mode_train == False:

            # plot the slice [:, :, 30]
            plt.figure("check", (18, 6))
            plt.subplot(1, 3, 1)
            plt.title(f"image {batch_idx}")
            plt.imshow(data[0, 0, :, :, 31], cmap="gray")
            plt.subplot(1, 3, 2)
            plt.title(f"label {batch_idx}")
            plt.imshow(y[0, 0, :, :, 31])
            plt.subplot(1, 3, 3)
            plt.title(f"output {batch_idx}")
            plt.imshow(torch.argmax(out, dim=1).detach().cpu()[0, :, :, 15])
            plt.show()

    return batch_loss.avg


def write(writer, epoch, loss):
    it = opt.epoch_init + epoch
    writer.add_scalar('Loss', loss, it)


def train(data_loader_train, data_loader_val, model, criterion, optimizer):
    # Writer
    writer_train = SummaryWriter('runs/' + model_name + '_train')
    writer_val = SummaryWriter('runs/' + model_name + '_val')

    best_val_loss = None
    counter = 0
    for epoch in range(opt.epoch):
        print("----------")
        print(f'Epoch: {epoch +1}/{opt.epoch}')

        # Train
        loss_train = train_epoch(data_loader_train, model, criterion, optimizer, mode_train=True)
        print(f'Training - Loss: {loss_train} ')
        write(writer_train, epoch, loss_train)

        # Validation
        loss_val = train_epoch(data_loader_val, model, criterion, mode_train=False)
        print(f'Validation - Loss: {loss_val}\n')
        write(writer_val, epoch, loss_val)

        # Early stopping
        if best_val_loss is None or best_val_loss > loss_val:
            best_val_loss = loss_val
            counter = 0
            # Save Unet
            if opt.save_models:
                torch.save(model, 'models_checkpoints/{}/best_model.pth'.format(model_name))
                print('New best Unet: epoch {}'.format(epoch))
        else:
            counter += 1
            if counter > opt.early_stopping:
                break

    # Close writer
    writer_train.close()
    writer_val.close()


def main():


    # Load Unet
    net = UNet()
    net = net.to(device)

    # load the init weight
    if opt.is_load:
        net.load_state_dict(torch.load(opt.load_path))

    # Transform:
    transform  = transforms.Compose(
        [transforms.Lambda(lambda x: custom_normalize(x,opt.mean,opt.sd))])

    # Train data
    data_train = PatchesDataset_2(opt.path,
                                 transform=transform,
                                 patch_dim=opt.patch_dim,
                                 num_patches = 2,
                                 mode='train',
                                 random_sampling=True)

    data_loader_train = DataLoader(data_train, batch_size=opt.batch,num_workers=4)

    # Validation data
    data_val =  PatchesDataset_2(opt.path,
                                 transform=None,
                                 patch_dim=opt.patch_dim,
                                 num_patches=1,
                                 mode='val',
                                 random_sampling=True)


    data_loader_val = DataLoader(data_val, batch_size=1, num_workers=4)

    # Loss function
    #loss = metrics.GeneralizedDiceLoss()
    criterion = GeneralizedDiceLoss(to_onehot_y=True, softmax=True)

    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=opt.lr)

    # Train
    train(data_loader_train, data_loader_val, net, criterion, optimizer)


def create_dir():
    if not os.path.exists('models_checkpoints'):
        os.makedirs('models_checkpoints')
    if not os.path.exists(os.path.join('models_checkpoints', model_name)):
        os.makedirs(os.path.join('models_checkpoints', model_name))


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start = time.time()
    create_dir()
    main()
    end = time.time()
    print('Total training time:', end - start)

    # Show example results

