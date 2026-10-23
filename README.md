#MDHGL
#We provide two folders:
DDAs: Code for the  binary drug-disease association prediction task (DDAs).
 DMAs: Code for the multi-class (3-class) drug-disease association prediction task (DMAs).
#You need to install the following packages first, with the same version as ours, to avoid errors:
- python >= 3.9
- torch == 2.2.0+cu118
- torch-geometric == 2.6.1
- torch-cluster == 1.6.3+pt22cu118
- torch-scatter == 2.1.2+pt22cu118
- torch-sparse == 0.6.18+pt22cu118
- torch-spline-conv == 1.2.2+pt22cu118
- scikit-learn == 1.6.1
- scipy == 1.13.1
#Preprocessed data is placed in the 'data/' folder
#Training model: python train.py
