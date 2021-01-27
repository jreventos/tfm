from tensorboardX import SummaryWriter
#tensorboard --logdir=runs
import argparse
import time
from model.Unet3D import *
from model.Unet import UNet
from dataset import*
from utils.patches import *

from preprocessing import custom_normalize

import mlflow.pytorch


from monai.losses import DiceLoss, GeneralizedDiceLoss

# Parser
parser = argparse.ArgumentParser()
# Set seed
# parser.add_argument("--seed", type=int, default=42, help="Seed")

# Data loader parameters
parser.add_argument("--path", type=str, default='patients_data',
                    help="Path to the training and val data")
parser.add_argument("--mean", type=list, default=21.201036, help="Mean of the data set MRI volumes")
parser.add_argument("--sd", type=list, default=40.294086, help="Standard deviation of the data set MRI volumes")
parser.add_argument("--patch_dim", type=list, default=[32, 32, 32], help="Patches dimensions")
#parser.add_argument("--stepsize", type=int, default=64, help="patches overlap stepsize")
parser.add_argument("--positive_probability", type=int, default=0.7, help="% of positive probability")

# Model parameters
parser.add_argument("--is_load", type=bool, default=True, help="weights initialization")
parser.add_argument("--load_path", type=str, default='models_checkpoints/Unet3D_epoch_900_batch_1_patchdim_[32, 32, 32]/checkpoint_900.pth', help="path to the weights net initialization")
parser.add_argument("--lr", type=int, default=0.0001, help="learning rate")


# In case of transfer models
# parser.add_argument("--feature_extract", type=bool, default=False, help="If true do features extraction, otherwise finetuning")
# parser.add_argument("--fully_connected", type=str, default=1, help="Type of fully connected after convolution")
# parser.add_argument("--fine_tuning_prop", type=str, default='complete', help='Proportion of the net unfreezed')

# Training parameters
parser.add_argument("--epoch", type=int, default=900, help="Number of epochs")
parser.add_argument("--epoch_init", type=int, default=0, help="Number of epochs where we want to initialize")
parser.add_argument("--batch", type=int, default=1, help="Number of examples in batch")
parser.add_argument("--verbose", type=int, default=10, help="Verbose, when we want to do a print")

parser.add_argument('--early_true', type=bool, default=False, help='If we want early stopping')
parser.add_argument("--early_stopping", type=int, default=10, help="Number of epochs without improvement to stop")
parser.add_argument("--save_models", type=bool, default=True, help='If true models will be saved')
opt = parser.parse_args()

# Define model name
model_name = '_'.join(['Unet3D','epoch', str(opt.epoch), 'batch', str(opt.batch),'patchdim',str(opt.patch_dim)])


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
        data = data.type(torch.cuda.FloatTensor)
        y = y.to(device)

        # Reset gradients
        if mode_train:
            optimizer.zero_grad()

        # Get prediction

        out = model(data.to(device))


        # print(out)
        # print(out.shape)

        # Get loss
        loss = criterion(out.to(device), y.to(device))


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

    return batch_loss.avg

def inference(data_loader, model, criterion):
    total_batch = len(data_loader)
    batch_loss = average_metrics()

    model.eval()

    for batch_idx, (data, y) in enumerate(data_loader):
        data = data.to(device)
        data = data.type(torch.cuda.FloatTensor)
        y = y.to(device)

        # Get prediction
        out = model(data.to(device))

        # print(out)
        print(out.shape)

        # Get loss
        loss = criterion(out.to(device), y.to(device))


        # Backpropagate error
        loss.backward()

        # Update loss

        batch_loss.update(loss.item())

        print(f'Iteration {(batch_idx + 1)}/{total_batch} - Loss: {batch_loss.val} ')



        # plot the slice [:, :, 25]
        plt.figure("check", (18, 6))
        plt.subplot(1, 3, 1)
        plt.title(f"image {batch_idx}")
        plt.imshow(data.detach().cpu()[0, 0, :, :, 25], cmap="gray")
        plt.subplot(1, 3, 2)
        plt.title(f"label {batch_idx}")
        plt.imshow(y.detach().cpu()[0, 0, :, :, 25])
        plt.subplot(1, 3, 3)
        plt.title(f"output {batch_idx}")
        plt.imshow(torch.argmax(out, dim=1).detach().cpu()[0, :, :, 25])
        plt.show()

        #DisplaySlices(data.detach().cpu()[0, 0, :, :, :], int(data.detach().cpu().max()))
        #DisplaySlices(torch.argmax(out, dim=1).detach().cpu()[0,:,:,:], int(1))



def write(writer, epoch, loss):
    it = opt.epoch_init + epoch
    writer.add_scalar('Loss', loss, it)


def train(data_loader_train, data_loader_val, model, criterion, optimizer, early_true):
    # Writer
    writer_train = SummaryWriter('runs/' + model_name + '_train')
    writer_val = SummaryWriter('runs/' + model_name + '_val')

    best_val_loss = None
    counter = 0
    for epoch in range(opt.epoch):
        print("----------")
        print(f'Epoch: {epoch + 1}/{opt.epoch}')

        # Train
        loss_train = train_epoch(data_loader_train, model, criterion, optimizer, mode_train=True)
        print(f'Training - Loss: {loss_train} ')
        write(writer_train, epoch, loss_train)
        mlflow.log_metric("Average_train_loss", loss_train, step=epoch)

        # Validation
        loss_val = train_epoch(data_loader_val, model, criterion, mode_train=False)
        print(f'Validation - Loss: {loss_val}\n')
        write(writer_val, epoch, loss_val)
        mlflow.log_metric("Average_val_loss", loss_val, step=epoch)

        # Early stopping
        if early_true:
            if best_val_loss is None or best_val_loss > loss_val:
                best_val_loss = loss_val
                counter = 0
            else:
                counter += 1
                if counter > opt.early_stopping:
                    break
        # Save Unet
        if opt.save_models and (epoch + 1) % opt.verbose == 0:
            torch.save(model, 'models_checkpoints/{}/checkpoint_{}.pth'.format(model_name,epoch+1))

    # Close writer
    writer_train.close()
    writer_val.close()

def tmp_function(x):
    custom_normalize(x,opt.mean,opt.sd)
def main():
    mlflow.set_experiment("LA_segmentation_complete")
    with mlflow.start_run():
        mlflow.log_param("Path", opt.load_path)
        mlflow.log_param("Num_epochs", opt.epoch)
        mlflow.log_param("Batch", opt.batch)
        mlflow.log_param("Patch_dim", opt.patch_dim)
        mlflow.log_param("Positive_probability", opt.positive_probability)
        mlflow.log_param("device", device)

        start = time.time()


        # Load the model
        if opt.is_load:
            # if inference
            #net.load_state_dict(torch.load(opt.load_path))
            net = torch.load(opt.load_path)
        else:
            # if training
            # net = UNet()
            net = UNet3D(in_channels=1, out_channels=2)
            net.to(device)
            mlflow.pytorch.save_model(net, 'Unet')


        # Transform:
        #transform = transforms.Compose( [transforms.Lambda(lambda x: custom_normalize(x, opt.mean, opt.sd))])
        #transform = transforms.Compose([transforms.Lambda(tmp_function)])


        data_train = BalancedPatchGenerator(opt.path,
                                            opt.patch_dim,
                                            positive_prob=opt.positive_probability,
                                            shuffle_images=False,
                                            mode='train',
                                            is_test=False,
                                            transform=None)

        data_loader_train = DataLoader(data_train, batch_size=opt.batch, shuffle=True,num_workers=4)

        # Validation data
        data_val = BalancedPatchGenerator(opt.path,
                                          (120, 120, 49),
                                          positive_prob=opt.positive_probability,
                                          shuffle_images=False,
                                          mode='val',
                                          is_test=False,
                                          transform=None)

        data_loader_val = DataLoader(data_val, batch_size=opt.batch, num_workers=4)

        # # Loss function
        # loss = metrics.GeneralizedDiceLoss()
        criterion = GeneralizedDiceLoss(to_onehot_y=True, softmax=True)

        # Optimizer
        optimizer = torch.optim.Adam(net.parameters(), 1e-4)

        # # Train
        if opt.is_load:
            inference(data_loader_val, net, criterion)
            end = time.time()
            print('Total inference time:', end - start)
        else:
            train(data_loader_train, data_loader_val, net, criterion, optimizer, opt.early_true)
            end = time.time()
            print('Total training time:', end - start)



def create_dir():
    if not os.path.exists('models_checkpoints'):
        os.makedirs('models_checkpoints')
    if not os.path.exists(os.path.join('models_checkpoints', model_name)):
        os.makedirs(os.path.join('models_checkpoints', model_name))


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.get_device_name())
    print(torch.cuda.is_available())
    create_dir()
    main()
