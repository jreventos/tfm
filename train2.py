from tensorboardX import SummaryWriter
#tensorboard --logdir=runs
import argparse
import math
from model.Unet3D import *
from model.Unet import UNet
from dataset import*
from utils.readers import *
from preprocessing import custom_normalize

import mlflow.pytorch
from vtk.util.numpy_support import *

from monai.losses import DiceLoss, GeneralizedDiceLoss
from monai.metrics import compute_meandice, compute_confusion_matrix_metric, get_confusion_matrix
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
import optuna

from BoundaryLoss import *

# Parser
parser = argparse.ArgumentParser()
# Set seed
# parser.add_argument("--seed", type=int, default=42, help="Seed")

# Data loader parameters
parser.add_argument("--path", type=str, default='Dataset_arrays',
                    help="Path to the training and val data")
parser.add_argument("--patch_dim", type=list, default=[60, 60, 32], help="Patches dimensions")
parser.add_argument("--positive_probability", type=int, default=0.7, help="% of positive probability")
parser.add_argument("--transforms", type=bool, default=True, help="transforms")

# Model parameters
parser.add_argument("--is_load", type=bool, default=False, help="weights initialization")
parser.add_argument("--load_path", type=str, default='models_checkpoints/ExtendedDataset_epoch_500_batch_1_patchdim_[60, 60, 32]_posprob_0.7_normalize_True_lr_0.0001_optim_Adam_GDL_8_GD_True_BL_True_alpha_0.1/checkpoint_500.pth', help="path to the weights net initialization")
parser.add_argument("--lr", type=int, default=0.0001, help="learning rate")



# Training parameters
parser.add_argument("--epoch", type=int, default=500, help="Number of epochs")
parser.add_argument("--epoch_init", type=int, default=0, help="Number of epochs where we want to initialize")
parser.add_argument("--batch", type=int, default=1, help="Number of examples in batch")
parser.add_argument("--verbose", type=int, default=10, help="Verbose, when we want to do a print")

parser.add_argument('--early_true', type=bool, default=False, help='If we want early stopping')
parser.add_argument("--early_stopping", type=int, default=5, help="Number of epochs without improvement to stop")
parser.add_argument("--save_models", type=bool, default=True, help='If true models will be saved')
opt = parser.parse_args()

# Define model name


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


def train_epoch(data_loader, model, criterion1, criterion2, optimizer=None, mode_train=True):
    total_batch = len(data_loader)
    batch_loss = average_metrics()

    batch_dice = average_metrics()
    post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=2)
    post_label = AsDiscrete(to_onehot=True, n_classes=2)

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


        # Evaluation
        outputs = post_pred(out.to(device))
        labels = post_label(y.to(device))
        dice_val = compute_meandice(y_pred=outputs, y=labels, include_background=False)

        if math.isnan(dice_val.item()):
            pass
        else:
            batch_dice.update(dice_val.item())

        # Losses
        if criterion1 and criterion2 == None:
            # Get REGION loss
            region_loss = criterion1(out.to(device), y.to(device))

            # Backpropagate error
            region_loss.backward()
            # Update loss
            batch_loss.update(region_loss.item())

        elif criterion1 == None and criterion2:

            # Get CONTOUR loss
            dist = one_hot2dist(y.detach().cpu().numpy())
            dist = torch.Tensor(dist)
            contour_loss = criterion2(out.to(device), dist.to(device))

            # Backpropagate error
            contour_loss.backward()
            # Update loss
            batch_loss.update(contour_loss.item())

        else:
            # Get REGION loss
            region_loss = criterion1(out.to(device), y.to(device))

            # Get CONTOUR loss
            dist = one_hot2dist(y.detach().cpu().numpy())
            dist = torch.Tensor(dist)
            contour_loss = criterion2(outputs.to(device), dist.to(device))

            # Combination both losses
            alpha = 0.3
            loss = (1-alpha)*region_loss + alpha*contour_loss
            # Backpropagate error
            loss.backward()
            # Update loss
            batch_loss.update(loss.item())

        # Optimize
        if mode_train:
            optimizer.step()

        # Log
        if (batch_idx + 1) % opt.verbose == 0 and mode_train:
            if criterion1 and criterion2 == None:
                print(f'Iteration {(batch_idx + 1)}/{total_batch} - GD Loss: {batch_loss.val} - Dice: {batch_dice.val}')
            elif criterion1 == None and criterion2:
                print(f'Iteration {(batch_idx + 1)}/{total_batch} - B Loss: {batch_loss.val} - Dice: {batch_dice.val}')
            else:
                print(f'Iteration {(batch_idx + 1)}/{total_batch} - GD & B Loss: {batch_loss.val} - Dice: {batch_dice.val}')

    return batch_loss.avg, batch_dice.avg

def generate_vtk_from_numpy(ndarray,filename):
    from tvtk.api import tvtk, write_data

    grid = tvtk.ImageData(spacing=(1.25, 1.25, 2.5), origin=(0, 0, 0),
                          dimensions=ndarray.shape)
    grid.point_data.scalars = ndarray.ravel(order='F')
    grid.point_data.scalars.name = 'scalars'

    # Writes legacy ".vtk" format if filename ends with "vtk", otherwise
    # this will write data using the newer xml-based format.
    write_data(grid, filename)


def inference(data_loader, model, criterion):
    total_batch = len(data_loader)
    batch_loss = average_metrics()
    dice, sensitivity, specificity, precision, f1_score, accuracy = average_metrics(), average_metrics(),average_metrics(),average_metrics(),average_metrics(),average_metrics()
    post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=2)
    post_label = AsDiscrete(to_onehot=True, n_classes=2)


    model.eval()

    for batch_idx, (data, y) in enumerate(data_loader):
        data = data.to(device)
        data = data.type(torch.cuda.FloatTensor)
        y = y.to(device)

        # Get prediction
        out = model(data.to(device))

        # Get loss
        loss = criterion(out.to(device), y.to(device))

        # Backpropagate error
        loss.backward()

        # Update loss
        batch_loss.update(loss.item())

        # Evaluation
        outputs = post_pred(out.to(device))
        labels = post_label(y.to(device))

        dice.update(compute_meandice(y_pred=outputs,  y=labels, include_background=False,).item())

        conf_matrix = get_confusion_matrix(y_pred=outputs,  y=labels, include_background=False,)
        sensitivity.update(compute_confusion_matrix_metric('sensitivity',conf_matrix).item())
        specificity.update(compute_confusion_matrix_metric('specificity', conf_matrix).item())
        precision.update(compute_confusion_matrix_metric('precision', conf_matrix).item())
        f1_score.update(compute_confusion_matrix_metric('f1 score', conf_matrix).item())
        accuracy.update(compute_confusion_matrix_metric('accuracy', conf_matrix).item())

        print(f'Iteration {(batch_idx + 1)}/{total_batch} - Loss: {batch_loss.val}  -Dice: {dice.val} ')

        # plot slice
        slice = out.shape[4]//2
        plt.figure("check", (18, 6))
        plt.subplot(1, 3, 1)
        plt.title(f"image {batch_idx+1}")
        plt.imshow(data.detach().cpu()[0, 0, :, :, slice], cmap="gray")
        plt.subplot(1, 3, 2)
        plt.title(f"label {batch_idx+1}")
        plt.imshow(y.detach().cpu()[0, 0, :, :, slice])
        plt.subplot(1, 3, 3)
        plt.title(f"output {batch_idx+1}")
        plt.imshow(torch.argmax(out, dim=1).detach().cpu()[0, :, :, slice])
        plt.show()
        #print(type(torch.argmax(out, dim=1).detach().cpu().numpy()[0,:,:,:]))
        #print(torch.argmax(out, dim=1).detach().cpu().numpy()[0,:,:,:].shape)

        y_predicted_array = torch.argmax(out, dim=1).detach().cpu().numpy().astype('float')[0,:,:,slice:]
        y_true_array = y.detach().cpu().numpy()[0, 0, :, :, slice:]
        mri_array = data.detach().cpu().numpy()[0, 0, :, :, slice:]

        file = f'patient_{batch_idx+1}.vtk'
        generate_vtk_from_numpy(y_predicted_array,'predicted_'+ file)
        generate_vtk_from_numpy(y_true_array,'true_'+file)
        generate_vtk_from_numpy(mri_array,'mri_'+file)


    print(f'Test loss:{batch_loss.avg} - Dice: {dice.avg}' )




def train(data_loader_train, data_loader_val, model, criterion1,criterion2, optimizer, early_true,model_name):

    average_train_loss = average_metrics()
    average_val_loss = average_metrics()
    average_train_dice = average_metrics()
    average_val_dice = average_metrics()

    best_val_loss = None
    best_val_dice = None
    counter = 0
    for epoch in range(opt.epoch):
        print("----------")
        print(f'Epoch: {epoch + 1}/{opt.epoch}')

        # Train
        loss_train, dice_train = train_epoch(data_loader_train, model, criterion1, criterion2, optimizer, mode_train=True)

        print(f'Training - Loss: {loss_train} - Dice: {dice_train}')
        mlflow.log_metric("Train_loss_evolution", loss_train, step=epoch)
        mlflow.log_metric("Train_dice_evolution", dice_train, step=epoch)

        average_train_loss.update(loss_train)
        average_train_dice.update(dice_train)

        mlflow.log_metric("Average_train_loss", average_train_loss.avg, step=epoch)
        mlflow.log_metric("Average_train_dice", average_train_dice.avg, step=epoch)

        # Validation
        loss_val, dice_val = train_epoch(data_loader_val, model, criterion1,criterion2, mode_train=False)
        print(f'Validation - Loss: {loss_val} - Dice: {dice_val} \n')

        # Early stopping
        #if early_true:
        if best_val_loss is None or best_val_loss > loss_val:
            best_val_loss = loss_val
        #    counter = 0
        #else:
        #    counter += 1
            #if counter > opt.early_stopping:
             #   break

        if best_val_dice is None or best_val_dice < dice_val:
            best_val_dice = dice_val

        # Save Unet
        if opt.save_models and (epoch + 1) % opt.verbose == 0:
            torch.save(model.state_dict(), 'models_checkpoints/{}/checkpoint_{}.pth'.format(model_name,epoch+1))


        mlflow.log_metric("Val_loss_evolution", loss_val, step=epoch)
        mlflow.log_metric("Val_dice_evolution", dice_val, step=epoch)
        average_val_loss.update(loss_val)
        average_val_dice.update(dice_val)
        mlflow.log_metric("Average_val_loss", average_val_loss.avg, step=epoch)
        mlflow.log_metric("Average_val_dice", average_val_dice.avg, step=epoch)
        mlflow.log_metric('Best_val_loss',best_val_loss,step=epoch)
        mlflow.log_metric('Best_val_dice', best_val_dice, step=epoch)


    return best_val_loss

def tmp_function(x):
    custom_normalize(x,opt.mean,opt.sd)

def objective(trial):

    mlflow.set_experiment("LA_segmentation_optuna_complete")
    with mlflow.start_run():
        # Get hyperparameter suggestions created by Optuna and log them as params using mlflow
        positive_probability, optimizer_name = suggest_hyperparameters(trial)
        lr = 1e-4
        positive_probability = 0.7
        # Create model
        model_name = '_'.join(
                        ['ExtendedDataset', 'epoch', str(opt.epoch), 'batch', str(opt.batch), 'patchdim', str(opt.patch_dim), 'posprob',
                        str(positive_probability), 'normalize', str(opt.transforms),'lr',str(lr),'optim',str(optimizer_name),
                         'GDL',str(8),'GD',str(True),'BL',str(True),'alpha',str(0.3)])
        create_dir(model_name)

        mlflow.log_param("Model_name", model_name)
        mlflow.log_param("Num_epochs", opt.epoch)
        mlflow.log_param("Batch", opt.batch)
        mlflow.log_param("Patch_dim", opt.patch_dim)
        mlflow.log_param("Positive_probability", positive_probability)
        mlflow.log_param("lr", lr)
        mlflow.log_param('Group Norm', str(8))
        mlflow.log_param("optimizer_name", optimizer_name)
        mlflow.log_param("device", device)


        # net = UNet()
        net = UNet3D(in_channels=1, out_channels=2)
        net.to(device)
        #mlflow.pytorch.save_model(net, 'Unet')


        # # Loss function
        # loss = metrics.GeneralizedDiceLoss()
        criterion1 = GeneralizedDiceLoss(to_onehot_y=True, softmax=True)
        criterion2 = BoundaryLoss(idc=[0])

        # Optimizer
        #optimizer = torch.optim.Adam(net.parameters(), 1e-4)


        if optimizer_name == "Adam":
            optimizer = torch.optim.Adam(net.parameters(), lr=lr)

        data_train = BalancedPatchGenerator(opt.path,
                                            opt.patch_dim,
                                            positive_prob=positive_probability,
                                            shuffle_images=True,
                                            mode='train',
                                            transform=opt.transforms)

        data_loader_train = DataLoader(data_train, batch_size=opt.batch, shuffle=True, num_workers=4)

        # Validation data
        data_val = BalancedPatchGenerator(opt.path,
                                          None,
                                          positive_prob=0.7,
                                          shuffle_images=False,
                                          mode='val',
                                          transform=None)

        data_loader_val = DataLoader(data_val, batch_size=opt.batch, num_workers=4)

        best_val_loss = train(data_loader_train, data_loader_val, net, criterion1,criterion2, optimizer, opt.early_true,model_name)

        return best_val_loss

        # Transform:
        #transform = transforms.Compose( [transforms.Lambda(lambda x: custom_normalize(x, opt.mean, opt.sd))])
        #transform = transforms.Compose([transforms.Lambda(tmp_function)])


def test():

    net = UNet3D(in_channels=1, out_channels=2)
    net.to(device)
    net.load_state_dict(torch.load(opt.load_path))
    #net = torch.load(opt.load_path)

    # Loss function
    # loss = metrics.GeneralizedDiceLoss()
    criterion = GeneralizedDiceLoss(to_onehot_y=True, softmax=True)

    # Test data
    data_test = BalancedPatchGenerator(opt.path,
                                       None,
                                       positive_prob=0.7,
                                       shuffle_images=False,
                                       mode='test',
                                       transform=None)

    data_loader_test = DataLoader(data_test, batch_size=opt.batch, num_workers=4)

    inference(data_loader_test, net, criterion)


def main():

    if opt.is_load:
        test()
    else:

        # Create the optuna study which shares the experiment name
        study = optuna.create_study(study_name="pytorch-mlflow-optuna", direction="minimize")
        study.optimize(objective, n_trials=1)

        # Print optuna study statistics
        print("\n++++++++++++++++++++++++++++++++++\n")
        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))


        print("Best trial:")
        trial = study.best_trial

        print("  Trial number: ", trial.number)
        print("  Loss (trial value): ", trial.value)

        print("  Params: ")

        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))


def suggest_hyperparameters(trial):
    # Learning rate on a logarithmic scale
    #lr = trial.suggest_float("lr", 1e-4,1e-4)
    # Dropout ratio in the range from 0.0 to 0.9 with step size 0.1
    positive_probability = trial.suggest_float("positive_probability", 0.5, 0.7,step=0.1)
    # Optimizer to use as categorical value
    optimizer_name = trial.suggest_categorical("optimizer_name", ["Adam"])

    return positive_probability, optimizer_name


def create_dir(model_name):
    if not os.path.exists('models_checkpoints'):
        os.makedirs('models_checkpoints')
    if not os.path.exists(os.path.join('models_checkpoints', model_name)):
        os.makedirs(os.path.join('models_checkpoints', model_name))


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.get_device_name())
    print(torch.cuda.is_available())
    main()
