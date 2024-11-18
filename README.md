## Data
Our data is here, https://www.kaggle.com/datasets/utkarshsaxenadn/animal-image-classification-dataset

## Schedule
https://docs.google.com/spreadsheets/d/1m4bQZRZKALrs-xQmP715Kf9ARu1MXARH8EMszwxCIg0/edit?usp=sharing

## CHTC GPU
https://chtc.cs.wisc.edu/uw-research-computing/gpu-jobs

## CHTC Pytorch
https://github.com/CHTC/templates-GPUs/tree/master/conda/pytorch-1.9

## File structure
The file structure is,
├── checkpoints/         
│   ├── model_epoch_xx.pth 
│   └── model_Final.pth    
├── datasets.py           
├── model.py              
├── train_parallel.py     
├── environments.yml      
└── README.md

## Train
First, please modify the path of train data and test data in the datasets.py

Second, please run python train_parallel.py

Third, please input the num_epoches and batch_size
