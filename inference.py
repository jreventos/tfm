
from model.Unet3D import *
from vtk.util.numpy_support import *
from monai.losses import DiceLoss, GeneralizedDiceLoss
from monai.metrics import compute_meandice, compute_hausdorff_distance
from monai.transforms import AsDiscrete
from dataset import *
from utils.average_metrics import *
import pandas as pd
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def inference(data_loader, model, criterion,model_name):

    total_batch = len(data_loader)
    test_loss = average_metrics()
    dice, hausdorff, sensitivity, specificity, precision, f1_score, accuracy = average_metrics(), average_metrics(),average_metrics(),average_metrics(),average_metrics(),average_metrics(), average_metrics()
    post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=2)
    post_label = AsDiscrete(to_onehot=True, n_classes=2)


    model.eval()
    results = []
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

        dice.update(compute_meandice(y_pred=outputs,  y=labels, include_background=False,).item())
        hausdorff.update(compute_hausdorff_distance(y_pred=outputs, y=labels, distance_metric='euclidean' ).item())

        print(f'Iteration {(batch_idx + 1)}/{total_batch} - Loss: {test_loss.val}  -Dice: {dice.val} - Hausdorff: {hausdorff.val} ')
        end = time.time()

        result = [batch_idx+1,test_loss.val,dice.val,hausdorff.val,end-start]
        results.append(result)


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

        y_predicted_array = torch.argmax(out, dim=1).detach().cpu().numpy().astype('float')[0,:,:,:]
        y_true_array = y.detach().cpu().numpy()[0, 0, :, :, :]
        mri_array = data.detach().cpu().numpy()[0, 0, :, :, :]

        file = f'patient_{batch_idx+1}.vtk'
        generate_vtk_from_numpy(y_predicted_array,'test_results/'+model_name + '/predicted_'+ file)
        generate_vtk_from_numpy(y_true_array,'test_results/'+model_name + '/true_'+ file)
        generate_vtk_from_numpy(mri_array,'test_results/'+model_name + '/mri_'+ file)




    print(f'Test loss:{test_loss.avg} - Dice: {dice.avg} - Hausdorff:{hausdorff.avg}' )
    end2 = time.time()
    results.append(['average',test_loss.avg,dice.avg,hausdorff.val,end2-start2])
    results_df = pd.DataFrame(results, columns=["test num", "Loss", "Dice", "Hausdorff",'Time'])
    results_df.to_csv('test_results/' + model_name + '/metrics_vals.csv', index=False)


def test():
    from main import opt
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

    if not os.path.exists('test_results/'+opt.load_path[:-4]):
        os.makedirs('test_results/'+opt.load_path[:-4])

    inference(data_loader_test, net, criterion,opt.load_path[:-4])
