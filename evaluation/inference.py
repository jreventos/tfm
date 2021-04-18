from torch.utils.data import DataLoader
from model.Unet3D import *
from vtk.util.numpy_support import *
from monai.losses import GeneralizedDiceLoss
from monai.metrics import compute_meandice, compute_hausdorff_distance
from monai.transforms import AsDiscrete
from training.dataset import *
from utils.average_metrics import *
import pandas as pd
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from processing.postprocessing import remove_small_objects

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def inference(data_loader, model, criterion,model_name):
    "Network ready for the inference step. Plots the predicted LA mask results together with the ground truth mask. "

    total_batch = len(data_loader)

    test_loss = average_metrics()


    dice, hausdorff = average_metrics(), average_metrics()

    # Softmax
    post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=2)
    post_label = AsDiscrete(to_onehot=True, n_classes=2)

    # set model eval mode!
    model.eval()
    results = []
    dice_collection, hasudorff_collection = [],[]
    start2 = time.time()
    for batch_idx, (data, y) in enumerate(data_loader):
        result = None
        start = time.time()
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
        test_loss.update(loss.item())

        # Evaluation
        outputs = post_pred(out.to(device))
        labels = post_label(y.to(device))

        # Post-processing
        post_processing = remove_small_objects(torch.argmax(out, dim=1).detach().cpu()[0, :, :, :])

        # Metrice
        dice.update(compute_meandice(y_pred=post_processing.to(device),  y=labels, include_background=False,).item())
        hausdorff.update(compute_hausdorff_distance(y_pred=post_processing.to(device), y=labels, distance_metric='euclidean' ).item())
        dice_collection.append(dice.val)
        hasudorff_collection.append(hausdorff.val)

        print(f'Iteration {(batch_idx + 1)}/{total_batch} - Loss: {test_loss.val}  -Dice: {dice.val} - Hausdorff: {hausdorff.val} ')
        end = time.time()

        result = [batch_idx+1,test_loss.val,dice.val,hausdorff.val,end-start]
        results.append(result)


        # PLOTS
        plt.figure("check", (10, 5))
        slice = out.shape[4]//2
        ground_truth = rotate(y.detach().cpu()[0, 0, :, :, slice], 90)
        mri_image = rotate(data.detach().cpu()[0, 0, :, :, slice], 90)
        predicted = rotate(torch.argmax(post_processing, dim=1).detach().cpu()[0, :, :, slice], 90)
        custom = matplotlib.colors.ListedColormap(['gray', 'red'])

        # LGE - MRI
        plt.subplot(1,3,1)
        plt.title(f'MRI test {batch_idx+1}')
        plt.axis('off')
        plt.imshow(mri_image, cmap="gray")

        # Ground truth mask + LGE-MRI
        plt.subplot(1,3,2)
        plt.axis('off')
        plt.title('Ground Truth')
        plt.imshow(mri_image, cmap="gray")
        plt.imshow(ground_truth, alpha=0.4,cmap=custom)

        # Predicted mask + LGE-MRI
        plt.subplot(1,3,3)
        plt.title(f'DSC:{round(dice.val,3)} - HD:{round(hausdorff.val,2)}')
        plt.axis('off')
        plt.imshow(mri_image, cmap="gray")
        plt.imshow(predicted, alpha=0.4,cmap=custom)
        plt.show()

        # Save mri, gt masks and mask predictions into VTK format
        y_predicted_array = torch.argmax(post_processing, dim=1).detach().cpu().numpy().astype('float')[0,:,:,:]
        y_true_array = y.detach().cpu().numpy()[0, 0, :, :, :]
        mri_array = data.detach().cpu().numpy()[0, 0, :, :, :]

        file = f'patient_{batch_idx+1}.vtk'
        generate_vtk_from_numpy(y_predicted_array,'test_results/'+model_name + '/predicted_'+ file)
        generate_vtk_from_numpy(y_true_array,'test_results/'+model_name + '/true_'+ file)
        generate_vtk_from_numpy(mri_array,'test_results/'+model_name + '/mri_'+ file)


    print(f'Test loss:{test_loss.avg} - Dice: {dice.avg} +/- {np.std(dice_collection)} - Hausdorff:{hausdorff.avg} +/- {np.std(hasudorff_collection)}' )
    end2 = time.time()

    # Save metric results into csv file
    results.append(['average',test_loss.avg,dice.avg,hausdorff.val,end2-start2])
    results_df = pd.DataFrame(results, columns=["test num", "Loss", "Dice", "Hausdorff",'Time'])
    results_df.to_csv('test_results/' + model_name + '/metrics_vals.csv', index=False)


def test():
    from main import opt
    net = UNet3D(in_channels=1, out_channels=2)
    net.to(device)
    net.load_state_dict(torch.load(opt.load_path))

    # Loss function
    criterion = GeneralizedDiceLoss(to_onehot_y=True, softmax=True)

    # Test data
    data_test = BalancedPatchGenerator(opt.path,
                                       None,
                                       positive_prob= None,
                                       shuffle_images= False,
                                       mode='test',
                                       transform=None,
                                       large=opt.large)

    data_loader_test = DataLoader(data_test, batch_size=opt.batch, num_workers=4)

    if not os.path.exists('test_results/'+opt.load_path[:-4]):
        os.makedirs('test_results/'+opt.load_path[:-4])

    inference(data_loader_test, net, criterion,opt.load_path[:-4])
