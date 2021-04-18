# Automatic Segmentation the Left Atrium from Late Gadolinium Enhancement 3D Magnetic Resonance Images
![img.png](images/img.png)
This is a Master Thesis Project conducted in the master degree of Artificial Intelligence (UPC, UB and URV) in collaboration with the Department of Arrhythmias from the Hospital Clínic de Barcelona. 

The main objective is to build a fully-automated architecture for the segmentation of the left atrium from Late Gadolinium Enhancement 3D MRI data. 
The model is a patch-based Unet architecture with region-based and contour-based losses. This code has been implemented using pytorch and inspired by the 3D Unet 
neural network by Özgün Çiçek et al. in the [3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation](https://arxiv.org/abs/1606.06650), and the boundary loss proposed by Kerdevec et al. in the 
[Boundary loss for highly unbalanced segmentation](https://arxiv.org/pdf/1812.07032.pdf). 


# ClínicLA dataset
The cardiac 3D LGE-MRI dataset used for this study belong to the Department of Arrhythmias of the Hospital Clínic de Barcelona. 
Images were collected for the study performed in the [Delayed  GadoliniumEnhancement Magnetic Resonance Imaging Detected Anatomic Gap Length in Wide Circumferential Pulmonary Vein Ablation Lesions Is Associated With Recurrence of Atrial Fibrillation](https://pubmed.ncbi.nlm.nih.gov/30562102/) 
and extended with other images obtained during clincal practice of ablation procedures.

The  dataset  consist  of  chest  view  LGE-MRI  images  from patients undergoing cardiac ablation procedures. 
The dataset cannot be shared due to legal and ethical concerns. 

# Train and evaluate the model
In order to use this code, CUDA toolkit must be installed within the environment. 
Before using this code the user must install the work packages stated in the "requirements.txt" file.

```bash
pip install -r requirements.txt
```
Then the user should change the parser parameters within the "main.py" file. Different parameters
must be changed for training or evaluating the model:

### Train 
First, the "is_load" parameter should be set to False:

```bash
parser.add_argument("--is_load", type=bool, default= False ,help="weights initialization")
```
Then, the user could change the model hyper-parameter, however, the actual confogiration 
is set up with the best combination of parameters after fine-tuning. Finally, the user has
to run the "main.py" file: 

```bash
python main.py 
```

### Evaluate
For evaluation mode, the "is_load" parameter should be set to True and the path to the model 
checpoint parameters has to be introduced in the "load_path":
```bash
parser.add_argument("--is_load", type=bool, default= False ,help="weights initialization")
parser.add_argument("--load_path", type=str, default=inference_path, help="path to the weights net initialization")
```

Finally, the user has to run the "main.py" file: 

```bash
python main.py 
```


