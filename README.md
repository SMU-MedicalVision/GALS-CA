# GALS-CE : AI-based LMs screening model with contrast-enhanced knowledge


<img src="https://github.com/SMU-MedicalVision/GALS-CE/blob/main/sample_png/Schematic%20illustration.png" width="400px">


# 1. Setup Environment
In order to run our model, we suggest you create a virtual environment
```
conda create -n GALS-CE_env python=3.8
```
and activate it with
```
conda activate GALS-CE_env
```
Subsequently, download and install the required libraries by running:
```
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```
# 2. Prepare the Dataset
To simplify the dataloading for your own dataset, we provide a default dataset that simply requires the path to the folder with your NifTI images inside, i.e.
```
./RAW_DATA     
├── ID_001                       
│        ├── NC.nii.gz             
│        ├── AP.nii.gz        
│        ├── PVP.nii.gz       
│        ├── DP.nii.gz        
│        ├── (Body_mask.nii.gz)  
│        ├── (Tumor_mask.nii.gz) 
│        └── (Liver_mask.nii.gz) 
├── ID_002
├── ... 
└── ID_N 
```



# 3. Training
- ## Quick Test (optional)
**Stage I**: Synthesis quick test
```
python ./main/train_GALS-CE_gen.py --gpu 0 --quick_test
```
**Stage II**: Identification quick test 
```
python ./main/train_GALS-CE_cla.py --gpu 0 --quick_test --gen_save_dir ./main/trained_models/GALS-CE_gen/{pred_*_...class_seg_time}/  
```
>{} should be changed to the actual path for saving the synthesis result.  


**Inference**(optional): quick test. After the training is completed, the inference will be automatically carried out. If you want to perform the inference separately, please run:
```
python ./main/train_GALS-CE_gen.py --gpu 0 --quick_test --inference_only --save_dir ./main/trained_models/GALS-CE_gen/{pred_*_...class_seg_time}/
python ./main/train_GALS-CE_cla.py --gpu 0 --quick_test --inference_only --gen_save_dir ./main/trained_models/GALS-CE_gen/{pred_*_...class_seg_time}/ --save_dir ./main/trained_models/GALS-CE_cla/{bs*_ImageSize*_epoch*_seed*_time}/
```

- ## Comprehensive Training
**Pretraining**:
```
python ./main/train_GALS-CE_pretrain.py --gpu 0
```
**Stage I**: First, you need to train the generation model. To do so in a prepared dataset, you can run the following command:
```
python ./main/train_GALS-CE_gen.py --gpu 0
```
**Stage II**: Second, you need to train the classification model by running the following command. 
```
python ./main/train_GALS-CE_cla.py --gpu 0 --gen_save_dir ./main/trained_models/GALS-CE_gen/{pred_*_...class_seg_time}/
```
>Note that you need to provide the path to the synthesis result to successfully run the command.


- ## Visualize the Training Process (optional)
You can use the following command to observe the loss curve of the training process, visualize the sample image, etc.
```
tensorboard --logdir ./main/trained_models/
```


[Supplement] Problem troubleshooting can be found in Error_troubleshooting.txt
# 4. Inference (optional)
After the training is completed, the inference will be automatically carried out. If you want to perform the inference separately, please run:
```
python ./main/train_GALS-CE_gen.py --gpu 0 --inference_only --save_dir ./main/trained_models/GALS-CE_gen/{pred_*_...class_seg_time}/
python ./main/train_GALS-CE_cla.py --gpu 0 --inference_only --gen_save_dir ./main/trained_models/GALS-CE_gen/{pred_*_...class_seg_time}/ --save_dir ./main/trained_models/GALS-CE_cla/{bs*_ImageSize*_epoch*_seed*_time}/
```

# Citation

To cite our work, please use
```
(To be updated)

```
