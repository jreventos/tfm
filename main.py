from torch.utils.data import DataLoader
from training.train import *
from evaluation.inference import *
from training.dataset import *
import mlflow.pytorch
import argparse
from vtk.util.numpy_support import *
from training.losses import *
import optuna


# Set device:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Create Parser:
parser = argparse.ArgumentParser()

# ============================ TRAINING / INFERENCE ===========================================
inference_path = "models_checkpoints_finals/Large_Increase_alpha_epoch_250_batch_1_patchdim_[90, 90, 32]_posprob_0.5_normalize_True_lr_0.0001_GDL_8_GD_True_BL_True/checkpoint_bes_val_dice151.pth"
parser.add_argument("--is_load", type=bool, default=True,help="weights initialization")
parser.add_argument("--load_path", type=str, default=inference_path, help="path to the weights net initialization")


# ============================ MODEL PARAMETERS  ===========================================
#Data loader parameters
parser.add_argument("--path", type=str, default='datasets/ClinicLA_dataset',help="Path to the training and val data")
parser.add_argument("--patch_dim", type=list, default=[90, 90, 32], help="Patches dimensions")
parser.add_argument("--positive_probability", type=int, default=0.5, help="% of positive probability")
parser.add_argument("--transforms", type=bool, default=True, help="transforms")
parser.add_argument("--large",type=bool, default=True, help="True: large ClinicLA_dataset, False: small ClinicLA_dataset")

# Training parameters
parser.add_argument("--lr", type=int, default=0.0001, help="learning rate")
parser.add_argument("--group_norm",type=int,default=8, help='number of group normalizations')
parser.add_argument("--epoch", type=int, default=250, help="Number of epochs")
parser.add_argument("--epoch_init", type=int, default=0, help="Number of epochs where we want to initialize")
parser.add_argument("--batch", type=int, default=1, help="Number of examples in batch")
parser.add_argument("--verbose", type=int, default=10, help="Verbose, when we want to do a print")
parser.add_argument("--region_loss",type=bool, default=True, help="Region loss (Generalized Dice Loss)")
parser.add_argument("--contour_loss",type=bool, default=True, help="Contour loss (Boundary Loss)")
parser.add_argument('--early_true', type=bool, default=False, help='If we want early stopping')
parser.add_argument("--early_stopping", type=int, default=5, help="Number of epochs without improvement to stop")
parser.add_argument("--save_models", type=bool, default=True, help='If true models will be saved')
opt = parser.parse_args()


# Mlflow setting:
def suggest_hyperparameters(trial):

    # Study Positive probability parameter
    positive = trial.suggest_float("patch_dimensions",0.3,0.5,step=0.2)

    return positive

def objective(trial):

    mlflow.set_experiment("TFM_jana_final_experiments")
    with mlflow.start_run():

        # Get hyperparameter suggestions created by Optuna and log them as params using mlflow
        positive_probability = suggest_hyperparameters(trial)
        positive_probability = 0.5

        model_name = '_'.join(
            ['Prova', 'epoch', str(opt.epoch), 'batch', str(opt.batch), 'patchdim', str(opt.patch_dim), 'posprob',
             str(positive_probability), 'normalize', str(opt.transforms), 'lr', str(opt.lr),
             'GDL', str(opt.group_norm), 'GD', str(opt.region_loss), 'BL', str(opt.contour_loss) ])

        parser_model = argparse.ArgumentParser()
        parser_model.add_argument("--model_name", type=str, default=model_name)

        opt_model = parser_model.parse_args()

        create_dir(opt_model.model_name)

        mlflow.log_param("Model_name", opt_model.model_name)
        mlflow.log_param("Num_epochs", opt.epoch)
        mlflow.log_param("Batch", opt.batch)
        mlflow.log_param("Patch_dim", opt.patch_dim)
        mlflow.log_param("Positive_probability", positive_probability)
        mlflow.log_param("lr", opt.lr)
        mlflow.log_param('Group Norm', opt.group_norm)
        mlflow.log_param("device", device)
        mlflow.log_param("large",opt.large)
        mlflow.log_param('alpha','increase')


        # Custom Unet
        net = UNet3D(in_channels=1, out_channels=2,num_groups=opt.group_norm)
        net.to(device)

        # Loss function
        criterion1, criterion2 = None, None

        if opt.region_loss:
            criterion1 = GeneralizedDiceLoss(to_onehot_y=True, softmax=True)

        if opt.contour_loss:
            criterion2 = BoundaryLoss(idc=[1])

        # Optimizer
        optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)

        # Train data loader
        data_train = BalancedPatchGenerator(opt.path,
                                            opt.patch_dim,
                                            positive_prob=positive_probability,
                                            shuffle_images=True,
                                            mode='train',
                                            transform=opt.transforms,large=opt.large)

        data_loader_train = DataLoader(data_train, batch_size=opt.batch, shuffle=True, num_workers=4)

        # Validation data loaders
        data_val = BalancedPatchGenerator(opt.path,
                                          None,
                                          positive_prob=None,
                                          shuffle_images=False,
                                          mode='val',
                                          transform=None, large= opt.large)

        data_loader_val = DataLoader(data_val, batch_size=opt.batch, num_workers=1)

        best_val_loss = train(data_loader_train, data_loader_val, net, criterion1,criterion2,None, optimizer, opt.early_true,opt_model.model_name)

        return best_val_loss



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


def create_dir(name):
    if not os.path.exists('models_checkpoints_finals/'):
        os.makedirs('models_checkpoints_finals/')
    if not os.path.exists(os.path.join('models_checkpoints_finals/', name)):
        os.makedirs(os.path.join('models_checkpoints_finals/', name))


if __name__ == '__main__':
    print(torch.cuda.get_device_name())
    print(torch.cuda.is_available())
    main()
