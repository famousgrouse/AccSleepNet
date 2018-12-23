# AccSleepNet #
A deep learning model for automatic sleep stage scoring based on sum vector magnitude of accelerometer 

Code for the model in the paper derived from DeepSleepNet: a Model for Automatic Sleep Stage Scoring based on Raw Single-Channel EEG by Akara Supratak, Hao Dong, Chao Wu, Yike Guo publication in [arXiv](https://arxiv.org/abs/1703.04046).

before start the program, please specify the dataset folder in train.py then run the script. 

## Environment ##
- Windows 10 64bit
- CUDA toolkit 9.0 and CuDNN v6
- Python 3.6
- [tensorflow-gpu (1.6+)]
- matplotlib
- scikit-learn
- scipy
- pandas
- may others

## Dataset ##
The first dataset is using V.T. van Hees et.al Estimating sleep parameters using an accelerometer without sleep diary

## Scoring sleep stages ##
Current scoring method use cross-validation fold.



## Get a summary ##
The train code will show a summary of the performance of AccSleepNet. The performance metrics are overall accuracy, per-class F1-score, and macro F1-score.




## ToDo ##
-Change the network to intake the accelerometer data 


## Licence ##
- For academic and non-commercial use only
- Apache License 2.0
