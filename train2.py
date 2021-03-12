from tensorboardX import SummaryWriter
#tensorboard --logdir=runs
import argparse
import math
from model.Unet3D import *
from model.Unet import UNet
from utils.readers import *
from utils.losses_metrics import *
from preprocessing import custom_normalize

import mlflow.pytorch
from vtk.util.numpy_support import *

from monai.losses import DiceLoss, GeneralizedDiceLoss
from monai.metrics import compute_meandice, compute_hausdorff_distance, compute_confusion_matrix_metric, get_confusion_matrix
from monai.transforms import AsDiscrete
import optuna
from main import opt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def train_epoch(data_loader, model, alpha, criterion1, criterion2, optimizer=None, mode_train=True):
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
                print(f'Iteration {(batch_idx + 1)}/{total_batch} - GD Loss: {batch_loss.val} - Dice: {batch_dice.val} - Hausdorff:{batch_hausdorff.val}')
            elif criterion1 == None and criterion2:
                print(f'Iteration {(batch_idx + 1)}/{total_batch} - B Loss: {batch_loss.val} - Dice: {batch_dice.val} -  Hausdorff:{batch_hausdorff.val}')
            else:
                print(f'Iteration {(batch_idx + 1)}/{total_batch} - GD & B Loss: {batch_loss.val} - Dice: {batch_dice.val} - Hausdorff:{batch_hausdorff.val}')

    return batch_loss.avg, batch_dice.avg, batch_hausdorff.avg



def train(data_loader_train, data_loader_val, model, criterion1,criterion2, optimizer, early_true,model_name):

    average_train_loss, average_val_loss  = average_metrics(), average_metrics()
    average_train_dice, average_val_dice  = average_metrics(), average_metrics()
    average_train_hausdorff, average_val_hausdorff = average_metrics(), average_metrics()


    best_val_loss, best_val_dice, best_val_hausdorff = None, None, None

    counter = 0
    alphas = np.logspace(-1, -0.6, opt.epoch)
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
        #    counter = 0
        #else:
        #    counter += 1
            #if counter > opt.early_stopping:
             #   break

        if best_val_dice is None or best_val_dice < dice_val:
            best_val_dice = dice_val

        if best_val_hausdorff is None or best_val_hausdorff < hausdorff_val:
            best_val_hausdorff = hausdorff_val

        mlflow.log_metric('Best_val_loss', best_val_loss, step=epoch)
        mlflow.log_metric('Best_val_dice', best_val_dice, step=epoch)
        mlflow.log_metric('Best_val_hausdorff', best_val_hausdorff, step=epoch)


        # Save Unet
        if opt.save_models and (epoch + 1) % opt.verbose == 0:
            torch.save(model.state_dict(), 'models_checkpoints_finals/{}/checkpoint_{}.pth'.format(model_name,epoch+1))


    return best_val_loss






def inference(data_loader, model, criterion):
    total_batch = len(data_loader)
    batch_loss = average_metrics()
    dice, hausdorff, sensitivity, specificity, precision, f1_score, accuracy = average_metrics(), average_metrics(),average_metrics(),average_metrics(),average_metrics(),average_metrics(), average_metrics()
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
        hausdorff.update(compute_hausdorff_distance(y_pred=outputs, y=labels, distance_metric='euclidean' ).item())

        # conf_matrix = get_confusion_matrix(y_pred=outputs,  y=labels, include_background=False,)
        # sensitivity.update(compute_confusion_matrix_metric('sensitivity',conf_matrix).item())
        # specificity.update(compute_confusion_matrix_metric('specificity', conf_matrix).item())
        # precision.update(compute_confusion_matrix_metric('precision', conf_matrix).item())
        # f1_score.update(compute_confusion_matrix_metric('f1 score', conf_matrix).item())
        # accuracy.update(compute_confusion_matrix_metric('accuracy', conf_matrix).item())

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


    print(f'Test loss:{batch_loss.avg} - Dice: {dice.avg} - Hausdorff:{batch_hausdorff.val}' )


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
                                       transform=None,large=opt.large)

    data_loader_test = DataLoader(data_test, batch_size=opt.batch, num_workers=4)

    inference(data_loader_test, net, criterion)






