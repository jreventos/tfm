
import math
from torch.nn.functional import softmax
import mlflow.pytorch
from vtk.util.numpy_support import *
from monai.metrics import compute_meandice, compute_hausdorff_distance
from monai.transforms import AsDiscrete
from main import opt
from training.dataset import *
from utils.average_metrics import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def train_epoch(data_loader, model, alpha, criterion1, criterion2, optimizer=None, mode_train=True):
    """
    Inputs:
    : data_loader: training or validation sets in dataset pytorch format
    : model (class): unet architecure
    : alpha (int): alpgha value for the boundary loss
    : criterion1: loss function 1 (in our case Generalized Dice Loss)
    : criterion2: loss function 2 ( in our case Boundary loss)
    : optimizer (class): define optimizer (ie. adam)
    : mode_train (bool): True is train, False is validation

    """
    total_batch = len(data_loader)
    batch_loss = average_metrics()
    batch_dice = average_metrics()
    batch_hausdorff = average_metrics()
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
        hausdorff_val = compute_hausdorff_distance(y_pred=outputs,y=labels,distance_metric='euclidean')

        if math.isnan(dice_val.item()) or math.isnan(hausdorff_val.item()):
            pass
        else:
            batch_dice.update(dice_val.item())
            batch_hausdorff.update(hausdorff_val.item())


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
            out_probs = softmax(out.to(device), dim=1)
            contour_loss = criterion2(out_probs.to(device), dy.to(device))

            # Backpropagate error
            contour_loss.backward()
            # Update loss
            batch_loss.update(contour_loss.item())

        else:
            # Get REGION loss
            region_loss = criterion1(out.to(device), y.to(device))

            # Get CONTOUR loss
            out_probs = softmax(out.to(device), dim=1)
            contour_loss = criterion2(out_probs.to(device), y.to(device))
            # Combination both losses
            loss = region_loss + alpha*contour_loss
            # Backpropagate error
            loss.backward()
            # Update loss
            # Update loss
            batch_loss.update(loss.item())

        # Optimize
        if mode_train:
            optimizer.step()

        # Log
        if (batch_idx + 1) % opt.verbose == 0 and mode_train:
            if criterion1 and criterion2 == None:
                print(f'Iteration {(batch_idx + 1)}/{total_batch} - GD Loss: {batch_loss.val} - Dice: {batch_dice.val} - Hausdorff:{batch_hausdorff.val}')
            elif criterion1 == None and criterion2:
                print(f'Iteration {(batch_idx + 1)}/{total_batch} - B Loss: {batch_loss.val} - Dice: {batch_dice.val} -  Hausdorff:{batch_hausdorff.val}')
            else:
                print(f'Iteration {(batch_idx + 1)}/{total_batch} - GD & B Loss: {batch_loss.val} - Dice: {batch_dice.val} - Hausdorff:{batch_hausdorff.val}')

    return batch_loss.avg, batch_dice.avg, batch_hausdorff.avg



def train(data_loader_train, data_loader_val, model, criterion1,criterion2, alpha, optimizer, early_true,model_name):

    """
    Inputs:
    : data_loader_train: training set in dataset pytorch format
    : data_loader_val: validation set in dataset pytorch format
    : model (class): unet architecure
    : alpha (int): alpgha value for the boundary loss (if its constant)
    : criterion1: loss function 1 (in our case Generalized Dice Loss)
    : criterion2: loss function 2 ( in our case Boundary loss)
    : optimizer (class): define optimizer (ie. adam)
    : early_true (bool): if true, the traininf performes early stopping strategy
    : model_name (string): define the model name according to the hyper parameters settings

    """

    average_train_loss, average_val_loss  = average_metrics(), average_metrics()
    average_train_dice, average_val_dice  = average_metrics(), average_metrics()
    average_train_hausdorff, average_val_hausdorff = average_metrics(), average_metrics()


    best_val_loss, best_val_dice, best_val_hausdorff = None, None, None

    counter = 0
    alphas = np.logspace(-1, -0.0001, opt.epoch)
    for epoch in range(opt.epoch):
        print("----------")
        print(f'Epoch: {epoch + 1}/{opt.epoch}')

        # Train
        alpha = alphas[epoch]
        loss_train, dice_train, hausdorff_train = train_epoch(data_loader_train, model, alpha, criterion1, criterion2, optimizer, mode_train=True)

        print(f'Training - Loss: {loss_train} - Dice: {dice_train} - Hausdorff: {hausdorff_train}')
        mlflow.log_metric("Train_loss_evolution", loss_train, step=epoch)
        mlflow.log_metric("Train_dice_evolution", dice_train, step=epoch)
        mlflow.log_metric("Train_hausdorff_evolution", hausdorff_train, step=epoch)

        average_train_loss.update(loss_train)
        average_train_dice.update(dice_train)
        average_train_hausdorff.update(hausdorff_train)

        mlflow.log_metric("Average_train_loss", average_train_loss.avg, step=epoch)
        mlflow.log_metric("Average_train_dice", average_train_dice.avg, step=epoch)
        mlflow.log_metric("Average_train_dice", average_train_dice.avg, step=epoch)

        # Validation
        loss_val, dice_val, hausdorff_val = train_epoch(data_loader_val, model, alpha, criterion1,criterion2, mode_train=False)
        print(f'Validation - Loss: {loss_val} - Dice: {dice_val} - Hausdorff: {hausdorff_val} \n')

        # Save validation in mlflow
        mlflow.log_metric("Val_loss_evolution", loss_val, step=epoch)
        mlflow.log_metric("Val_dice_evolution", dice_val, step=epoch)
        mlflow.log_metric("Val_hausdorff_evolution", hausdorff_val, step=epoch)

        average_val_loss.update(loss_val)
        average_val_dice.update(dice_val)
        average_val_hausdorff.update(hausdorff_val)

        mlflow.log_metric("Average_val_loss", average_val_loss.avg, step=epoch)
        mlflow.log_metric("Average_val_dice", average_val_dice.avg, step=epoch)
        mlflow.log_metric("Average_val_hausdorff", average_val_hausdorff.avg, step=epoch)



        # Early stopping
        #if early_true:
        if best_val_loss is None or best_val_loss > loss_val:
            best_val_loss = loss_val
            torch.save(model.state_dict(),'models_checkpoints_finals/{}/checkpoint_bes_val_loss{}.pth'.format(model_name, epoch + 1))
        #    counter = 0
        #else:
        #    counter += 1
            #if counter > opt.early_stopping:
             #   break

        if best_val_dice is None or best_val_dice < dice_val:
            best_val_dice = dice_val
            torch.save(model.state_dict(),'models_checkpoints_finals/{}/checkpoint_bes_val_dice{}.pth'.format(model_name,epoch + 1))

        if best_val_hausdorff is None or best_val_hausdorff > hausdorff_val:
            best_val_hausdorff = hausdorff_val
            torch.save(model.state_dict(),'models_checkpoints_finals/{}/checkpoint_bes_val_hausdorff{}.pth'.format(model_name,epoch + 1))

        mlflow.log_metric('Best_val_loss', best_val_loss, step=epoch)
        mlflow.log_metric('Best_val_dice', best_val_dice, step=epoch)
        mlflow.log_metric('Best_val_hausdorff', best_val_hausdorff, step=epoch)


        # Save Unet
        if opt.save_models and (epoch + 1) % opt.verbose == 0:
            torch.save(model.state_dict(), 'models_checkpoints_finals/{}/checkpoint_{}.pth'.format(model_name,epoch+1))

    return best_val_loss

