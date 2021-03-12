
from train2 import *
from dataset import *
from model import Unet3D
import mlflow.pytorch
import argparse
from model.Unet3D import *

from utils.losses_metrics import *
from vtk.util.numpy_support import *
from monai.losses import DiceLoss, GeneralizedDiceLoss

import optuna
from optuna.pruners import BasePruner


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parser
parser = argparse.ArgumentParser()
# Set seed
# parser.add_argument("--seed", type=int, default=42, help="Seed")

# Data loader parameters
parser.add_argument("--path", type=str, default='dataset_patients',
                    help="Path to the training and val data")
parser.add_argument("--patch_dim", type=list, default=[90, 90, 32], help="Patches dimensions")
parser.add_argument("--positive_probability", type=int, default=0.7, help="% of positive probability")
parser.add_argument("--transforms", type=bool, default=True, help="transforms")
parser.add_argument("--large",type=bool, default=False, help="True: large dataset_patients, False: small dataset_patients")

# Model parameters
parser.add_argument("--is_load", type=bool, default=False, help="weights initialization")
parser.add_argument("--load_path", type=str, default='models_checkpoints/ExtendedDataset_Unet3D_epoch_500_batch_1_patchdim_[60, 60, 32]_posprob_0.7_normalize_True_lr_0.0001_optim_Adam_GN_8/checkpoint_500.pth', help="path to the weights net initialization")
parser.add_argument("--lr", type=int, default=0.0001, help="learning rate")
parser.add_argument("--group_norm",type=int,default=8, help='number of group normalizations')


# Training parameters
parser.add_argument("--epoch", type=int, default=250, help="Number of epochs")
parser.add_argument("--epoch_init", type=int, default=0, help="Number of epochs where we want to initialize")
parser.add_argument("--batch", type=int, default=1, help="Number of examples in batch")
parser.add_argument("--verbose", type=int, default=10, help="Verbose, when we want to do a print")
parser.add_argument("--region_loss",type=bool, default=True, help="Region loss (Generalized Dice Loss)")
parser.add_argument("--contour_loss",type=bool, default=False, help="Contour loss (Boundary Loss)")


parser.add_argument('--early_true', type=bool, default=False, help='If we want early stopping')
parser.add_argument("--early_stopping", type=int, default=5, help="Number of epochs without improvement to stop")
parser.add_argument("--save_models", type=bool, default=True, help='If true models will be saved')

opt = parser.parse_args()

def suggest_hyperparameters(trial):
    # Learning rate on a logarithmic scale
    #lr = trial.suggest_float("lr", 1e-4,1e-4)
    # Dropout ratio in the range from 0.0 to 0.9 with step size 0.1
    positive_probability = trial.suggest_float("positive_probability", 0.3, 0.8,step=0.1)
    # Optimizer to use as categorical value
    #patch_dim = trial.suggest_float("patch_dimensions",30,60,step=30)


    return round(positive_probability,1)

def objective(trial):

    mlflow.set_experiment("TFM_jana_final_experiments")
    with mlflow.start_run():

        # Get hyperparameter suggestions created by Optuna and log them as params using mlflow
        positive_probability = suggest_hyperparameters(trial)


        print(positive_probability)

        model_name = '_'.join(
            ['Small', 'epoch', str(opt.epoch), 'batch', str(opt.batch), 'patchdim', str(opt.patch_dim), 'posprob',
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


        # net = UNet()
        net = UNet3D(in_channels=1, out_channels=2)
        net.to(device)
        #mlflow.pytorch.save_model(net, 'Unet')


        # # Loss function
        # loss = metrics.GeneralizedDiceLoss()
        criterion1, criterion2 = None, None

        if opt.region_loss:
            criterion1 = GeneralizedDiceLoss(to_onehot_y=True, softmax=True)

        if opt.contour_loss:
            criterion2 = BoundaryLoss(idc=[0])

        # Optimizer
        optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)

        data_train = BalancedPatchGenerator(opt.path,
                                            opt.patch_dim,
                                            positive_prob=positive_probability,
                                            shuffle_images=True,
                                            mode='train',
                                            transform=opt.transforms,large=opt.large)



        data_loader_train = DataLoader(data_train, batch_size=opt.batch, shuffle=True, num_workers=4)

        # Validation data
        data_val = BalancedPatchGenerator(opt.path,
                                          None,
                                          positive_prob=positive_probability,
                                          shuffle_images=False,
                                          mode='val',
                                          transform=None, large= opt.large)

        data_loader_val = DataLoader(data_val, batch_size=opt.batch, num_workers=4)

        best_val_loss = train(data_loader_train, data_loader_val, net, criterion1,criterion2, optimizer, opt.early_true,opt_model.model_name)

        return best_val_loss

class RepeatPruner(BasePruner):
    def prune(self, study, trial):
        # type: (Study, FrozenTrial) -> bool

        trials = study.get_trials(deepcopy=False)
        completed_trials = [t.params for t in trials if t.state == TrialState.COMPLETE]
        n_trials = len(completed_trials)

        if n_trials == 0:
            return False

        if trial.params in completed_trials:
            return True

        return False


def main():

    if opt.is_load:
        test()
    else:

        # Create the optuna study which shares the experiment name
        study = optuna.create_study(study_name="pytorch-mlflow-optuna", direction="minimize",pruner=RepeatPruner())
        study.optimize(objective, n_trials=6)

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
    if not os.path.exists('models_checkpoints_finals'):
        os.makedirs('models_checkpoints_finals')
    if not os.path.exists(os.path.join('models_checkpoints_finals', name)):
        os.makedirs(os.path.join('models_checkpoints_finals', name))


if __name__ == '__main__':
    print(torch.cuda.get_device_name())
    print(torch.cuda.is_available())
    main()
