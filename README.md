# GenCAD 3D

---
![Logo](https://github.com/yunomi-git/GenCAD-3D/blob/main/GenCAD3D_Icon.png)

Official implementation of the paper **GenCAD3D: CAD Program Generation through Multimodal Latent Space Alignment and Synthetic Dataset Balancing**

**Project Page**: https://gencad3d.github.io/

**Code**: https://github.com/yunomi-git/GenCAD-3D

**Dataset**: https://huggingface.co/datasets/yu-nomi/GenCAD_3D

**Weights**: https://huggingface.co/yu-nomi/GenCAD_3D

---
## Download Datasets
Download the datasets from https://huggingface.co/datasets/yu-nomi/GenCAD_3D to your desired location "data_path/". Then unzip.

There are 4 datasets:
    
    GenCAD3D: 127 GB [CAD programs, clouds, meshes, stls, steps]
    GenCAD3D_Scans: 700 MB [raw scans, clean scans, CAD programs, clouds, meshes]
    GenCAD3D_SynthBal: 109 GB [CAD programs, clouds, meshes, stls]
    GenCAD3D_SynthBal_1M: 9 GB [CAD programs]

Then update the file paths.py by setting
    
    # paths.py   
    DATA_PATH = <data_path/>

## Download Weights

Download weights from https://huggingface.co/yu-nomi/GenCAD_3D with the following folder structure

    ./results
        Autoencoder/
            ...
        Contrastive/
            ...
        Diffusion/
            ...

## Environment
    
    pip install torch torchvision torchaudio   
    # This repo was tested with torch 2.7.1
    pip install -r requirements.txt
    mamba install -c conda-forge pythonocc-core=7.9.0

---
# Quickstart Inference

If you want to directly generate CAD programs with our model, use

    ## Visualize on your own stls
    python -m diffusion.evaluation.visualize_diffusion_inference -encoder_type mesh_feast -contrastive_model_name mesh_SynthBal_1M_SBD -filenames examples_files/00152170.stl

    ## Visualize on the GenCAD3D dataset
    python -m diffusion.evaluation.visualize_diffusion_samples -encoder_type pc -contrastive_model_name pcn_SynthBal_1M_SBD 
    # -contrastive_model_name: select from pc_SynthBal_1M_SBD, pcn_SynthBal_1M_SBD, mesh_SynthBal_1M_SBD 
    # -encoder_type: choose pc or mesh_feast
    # This visualizes assorted samples from the GenCAD3D dataset listed in custom_visualization_sets.py

## Translation to CAD Software

Open an Onshape account at http://onshape.com/

Set up an [API key](https://onshape-public.github.io/docs/auth/apikeys/), and place the information into a file in the main directory:

    <onshape_key.json>
    {"secret_key":  "xxx",
    "access_key":  "xxx"}

Create a new part studio, then copy in the url and run

    python -m GenCADGenerator.program_to_cad -url https://cad.onshape.com/d/w/e.com -cad_path example_files/00152170_gen_cad.h5

---
# Training and Testing Pipeline
## Autoencoder
### Train 
    # For general training
    python -m autoencoder.gencad.train_gencad -name autoencoder_model -epoch 1000 -lr 5e-3 -b 512 -gpu 0 -sf 100 --warm_up 2000 

    # For SynthBal Training
    python -m autoencoder.gencad.train_gencad -name autoencoder_model_SB -epoch 1000 -lr 5e-3 -b 512 -gpu 0 -sf 100 --warm_up 2000 --data_root GenCAD3D_SynthBal/
    python -m autoencoder.gencad.train_gencad -name autoencoder_model_SB_FT -epoch 100 -lr 5e-4 -b 512 -gpu 0 -sf 50 --fine_tune --warm_up 200 --data_root GenCAD3D_SynthBal/ -ckpt results/autoencoder_example_SB/autoencoder/trained_models/latest.pth 

### Test
    python -m autoencoder.evaluation.run_autoencoder_reconstruction -name autoencoder_model --subfolder 1000 -ckpt latest
    # add --cd to evaluate chamfer distance and invalid ratio. To perform this evaluations, first generate point cloud samples according to the steps in https://github.com/rundiwu/DeepCAD
    # add --iou to evaluate iou
    # add -gen_step if it is your first evaluation of iou

### (Optional: Create Custom SynthBal)
If you want to recreate our SynthBal dataset, or other custom datasets, you can use 
    
    python -m autoencoder.synthbal.generate_augmented_dataset -name GenCAD3D_SynthBal --input_dataset GenCAD3D
    python -m autoencoder.synthbal.combine_augmented_dataset -name GenCAD3D_SynthBal
    rm -r data_path/
    python -m autoencoder.synthbal.create_synthetic_splits -name GenCAD3D_SynthBal --input_dataset GenCAD3D

---
## Contrastive

### Train
    python -m contrastive.train_contrastive_model -encoder_type pc -name pcn_contrastive_model -num_workers 24 -use_normals --autoencoder_model_name Autoencoder_SynthBal_1MFT -dataset GenCAD3D
    # -encoder_type can be pc or mesh_feast
    # -use_normals for using normals
    # -autoencoder_model_name to select a CAD autoencoder model to create encodings
    # -dataset to select the dataset to train this model with, for example GenCAD3D or GenCAD3D_SynthBal
    # -bs batch size

### Test
    python -m contrastive.evaluation.eval_contrastive_retrieval -encoder_type pc -contrastive_model_name pcn_contrastive_model -checkpoint latest
    # -eval_method
    #   "accuracy": performs top-n retrieval and returns accuracy
    #   "images": plots images of retrieval

---
## Diffusion

### Train
    python -m diffusion.generate_diffusion_embeddings -encoder_type pc -contrastive_model_name pc_contrastive_model -checkpoint 300
    python -m diffusion.train_cond_diffusion -num_workers 0 -encoder_type pc -contrastive_model_name pc_contrastive_model -checkpoint 300 
    
    # -dataset to select the dataset to train this model with, for example GenCAD3D or GenCAD3D_SynthBal
### Test
    ## Visualization
    python -m diffusion.evaluation.visualize_diffusion_samples -encoder_type pc -contrastive_model_name pc_contrastive_model -contrastive_checkpoint latest -diffusion_checkpoint latest

    ## Generation Metrics
    python -m diffusion.evaluation.run_generation_evaluations -encoder_type pc -contrastive_model_name pc_contrastive_model -contrastive_checkpoint latest -diffusion_checkpoint latest

    ## Reconstruction Metrics
    python -m diffusion.evaluation.run_reconstruction_evaluations -encoder_type pc -contrastive_model_name pc_contrastive_model -contrastive_checkpoint latest -diffusion_checkpoint latest
    # add --cd to evaluate chamfer distance and invalid ratio
    # add --iou to evaluate iou
    # add -gen_step if it is your first evaluation of iou


