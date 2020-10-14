# LA Segmentation 

This is a Master Thesis Project conducted in the master of Artificial Intelligence (UPC, UB and URV) in collaboration with the department of Arrhythmia of the Hospital Clínic de Barcelona. 
The main objective is the implementation of a fully-automated architecture for the segmentation of the left atrium from Late Gadolinium Enhancement 3D MRI data.

This code is a implementation of a pytorch 3D Unet inspired by by Özgün Çiçek et al. [3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation](https://arxiv.org/abs/1606.06650).

# Dataset
The cardiac MRI dataset used for this study belong to the Arryhtmia department of the Hospital Clínic de Barcelona. 
Images were collected for the study performed in the [Delayed  GadoliniumEnhancement Magnetic Resonance Imaging Detected Anatomic Gap Length in Wide Circumferen-tial Pulmonary Vein Ablation Lesions Is Associated With Recurrence of Atrial Fibrillation](https://pubmed.ncbi.nlm.nih.gov/30562102/).   
The  dataset  consist  of  chest  view  MRI  images  from 85 patients in VTK binary or ascii image format. The full dataset cannot be shared here for legal reasons. 


# TODO: 
- Patches implementation into the dataloader (study which is the best approach to proceed: one patch per patient, X random patches per patient...)
- Discuss final volume dimensions (crop volumes according to the masks boundaries)
- Discuss weights initialization 
- Metrics
- Train file 
