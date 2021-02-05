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
from vtk.util.numpy_support import *

from monai.losses import DiceLoss, GeneralizedDiceLoss
import optuna

# Parser
parser = argparse.ArgumentParser()
# Set seed
# parser.add_argument("--seed", type=int, default=42, help="Seed")

# Data loader parameters
parser.add_argument("--path", type=str, default='patients_data',
                    help="Path to the training and val data")
parser.add_argument("--mean", type=list, default=21.201036, help="Mean of the data set MRI_volumes volumes")
parser.add_argument("--sd", type=list, default=40.294086, help="Standard deviation of the data set MRI_volumes volumes")
parser.add_argument("--patch_dim", type=list, default=[32, 32, 32], help="Patches dimensions")
parser.add_argument("--positive_probability", type=int, default=0.7, help="% of positive probability")
parser.add_argument("--transforms", type=bool, default=True, help="transforms")

# Model parameters
parser.add_argument("--is_load", type=bool, default=True, help="weights initialization")
parser.add_argument("--load_path", type=str, default='models_checkpoints/Unet3D_epoch_1000_batch_1_patchdim_[32, 32, 32]_posprob_0.6_normalize_True_lr_0.0001_optim_Adam/checkpoint_860.pth', help="path to the weights net initialization")
parser.add_argument("--lr", type=int, default=0.0001, help="learning rate")



# Training parameters
parser.add_argument("--epoch", type=int, default=1000, help="Number of epochs")
parser.add_argument("--epoch_init", type=int, default=0, help="Number of epochs where we want to initialize")
parser.add_argument("--batch", type=int, default=1, help="Number of examples in batch")
parser.add_argument("--verbose", type=int, default=20, help="Verbose, when we want to do a print")

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
        print(type(torch.argmax(out, dim=1).detach().cpu().numpy()[0,:,:,:]))
        print(torch.argmax(out, dim=1).detach().cpu().numpy()[0,:,:,:].shape)
        out_array = torch.argmax(out, dim=1).detach().cpu().numpy()[0,:,:,:]

        output_file = f'out_vtk_'+str(batch_idx)+'.vtk'
        vtk_out = convert2(out_array,output_file)
        print(type(vtk_out))


        # import pyvista as pv
        # data = pv.wrap(narray)
        # print(type(data))
        # data.plot(volume=True)

        #vtk_out = numpy_to_vtk(torch.argmax(out, dim=1).detach().cpu()[0,:,:,:])
        #DisplaySlices(data.detach().cpu()[0, 0, :, :, :], int(data.detach().cpu().max()))
        #DisplaySlices(torch.argmax(out, dim=1).detach().cpu()[0,:,:,:], int(1))



import SimpleITK as sitk



# Convert numpy array to VTK array (vtkFloatArray)
def convert2(ndarray,outputfile):
    array_tarans = ndarray.transpose(2, 1, 0)
    spacings = [(array_tarans[:,0,0].max() -array_tarans[:,0,0].min() ) / (array_tarans.shape[0] - 1), \
               (array_tarans[0,:,0].max() -array_tarans[0,:,0].min() ) / (array_tarans.shape[1] - 1), \
                (array_tarans[0,0,:].max() - array_tarans[0,0,:].min()) / (array_tarans.shape[2] - 1)]

    print(spacings)

    vtk_data_array = numpy_support.numpy_to_vtk(
        num_array=ndarray.transpose(2, 1, 0).ravel(),  # ndarray contains the fitting result from the points. It is a 3D array
        deep=True,
        array_type=vtk.VTK_FLOAT)


    # Convert the VTK array to vtkImageData
    img_vtk = vtk.vtkImageData()
    img_vtk.SetDimensions(ndarray.shape)
    img_vtk.SetSpacing(spacings[0],spacings[1],spacings[2])
    img_vtk.GetPointData().SetScalars(vtk_data_array)


    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(outputfile)
    if vtk.VTK_MAJOR_VERSION <= 5:
        writer.SetInputConnection(img_vtk.GetProducerPort())
    else:
        writer.SetInputData(img_vtk)
    writer.Write()

    return img_vtk


def write(writer, epoch, loss):
    it = opt.epoch_init + epoch
    writer.add_scalar('Loss', loss, it)


def train(data_loader_train, data_loader_val, model, criterion, optimizer, early_true,model_name):
    # Writer
    writer_train = SummaryWriter('runs/' + model_name + '_train')
    writer_val = SummaryWriter('runs/' + model_name + '_val')
    average_train_loss = average_metrics()
    average_val_loss = average_metrics()
    best_val_loss = None
    counter = 0
    for epoch in range(opt.epoch):
        print("----------")
        print(f'Epoch: {epoch + 1}/{opt.epoch}')

        # Train
        loss_train = train_epoch(data_loader_train, model, criterion, optimizer, mode_train=True)
        print(f'Training - Loss: {loss_train} ')
        write(writer_train, epoch, loss_train)

        mlflow.log_metric("Train_loss_evolution", loss_train, step=epoch)
        average_train_loss.update(loss_train)
        mlflow.log_metric("Average_train_loss", average_train_loss.avg, step=epoch)

        # Validation
        loss_val = train_epoch(data_loader_val, model, criterion, mode_train=False)
        print(f'Validation - Loss: {loss_val}\n')
        write(writer_val, epoch, loss_val)


        # Early stopping
        #if early_true:
        if best_val_loss is None or best_val_loss > loss_val:
            best_val_loss = loss_val
        #    counter = 0
        #else:
        #    counter += 1
            #if counter > opt.early_stopping:
             #   break

        # Save Unet
        if opt.save_models and (epoch + 1) % opt.verbose == 0:
            torch.save(model.state_dict(), 'models_checkpoints/{}/checkpoint_{}.pth'.format(model_name,epoch+1))


        mlflow.log_metric("Val_loss_evolution", loss_val, step=epoch)
        average_val_loss.update(loss_val)
        mlflow.log_metric("Average_val_loss", average_val_loss.avg, step=epoch)
        mlflow.log_metric('Best_val_loss',best_val_loss,step=epoch)

    # Close writer
    writer_train.close()
    writer_val.close()

    return best_val_loss

def tmp_function(x):
    custom_normalize(x,opt.mean,opt.sd)

def objective(trial):

    mlflow.set_experiment("LA_segmentation_optuna_complete")
    with mlflow.start_run():
        # Get hyperparameter suggestions created by Optuna and log them as params using mlflow
        _, optimizer_name = suggest_hyperparameters(trial)
        lr = 1e-4
        positive_probability = 0.6
        # Create model
        model_name = '_'.join(
                        ['Unet3D', 'epoch', str(opt.epoch), 'batch', str(opt.batch), 'patchdim', str(opt.patch_dim), 'posprob',
                        str(positive_probability), 'normalize', str(opt.transforms),'lr',str(lr),'optim',str(optimizer_name)])
        create_dir(model_name)

        mlflow.log_param("Model_name", model_name)
        mlflow.log_param("Num_epochs", opt.epoch)
        mlflow.log_param("Batch", opt.batch)
        mlflow.log_param("Patch_dim", opt.patch_dim)
        mlflow.log_param("Positive_probability", positive_probability)
        mlflow.log_param("lr", lr)
        mlflow.log_param("optimizer_name", optimizer_name)
        mlflow.log_param("device", device)


        # net = UNet()
        net = UNet3D(in_channels=1, out_channels=2)
        net.to(device)
        #mlflow.pytorch.save_model(net, 'Unet')


        # # Loss function
        # loss = metrics.GeneralizedDiceLoss()
        criterion = GeneralizedDiceLoss(to_onehot_y=True, softmax=True)

        # Optimizer
        #optimizer = torch.optim.Adam(net.parameters(), 1e-4)


        if optimizer_name == "Adam":
            optimizer = torch.optim.Adam(net.parameters(), lr=lr)

        data_train = BalancedPatchGenerator(opt.path,
                                            opt.patch_dim,
                                            positive_prob=positive_probability,
                                            shuffle_images=False,
                                            mode='train',
                                            transform=True)

        data_loader_train = DataLoader(data_train, batch_size=opt.batch, shuffle=True, num_workers=4)

        # Validation data
        data_val = BalancedPatchGenerator(opt.path,
                                          (120, 120, 49),
                                          positive_prob=positive_probability,
                                          shuffle_images=False,
                                          mode='val',
                                          transform=None)

        data_loader_val = DataLoader(data_val, batch_size=opt.batch, num_workers=4)

        best_val_loss = train(data_loader_train, data_loader_val, net, criterion, optimizer, opt.early_true,model_name)

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

    positive_probability = 0.7
    data_test = BalancedPatchGenerator(opt.path,
                                       (120, 120, 49),
                                       positive_prob=positive_probability,
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
    positive_probability = trial.suggest_float("positive_probability", 0.7, 0.9,step=0.1)
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
