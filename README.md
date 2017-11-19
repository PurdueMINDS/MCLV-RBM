# MCLV-RBM
Training RBMs using MCLV

# Setting up the code

You will need Anaconda
Create the working directory  and navigate into it

```
unzip MCLV.zip
```

You'll see the following folder structure

```
LICENSE  MCLV.zip  py  README.MD  results
```

Creating the required conda environment

```
conda create -n mclv_env python=3.6 pip

source activate mclv_env

pip install --upgrade pip

pip install numpy

pip install pandas

pip install scipy

pip install matplotlib

pip install sklearn

pip install seaborn
```

For pytorch please refer to http://pytorch.org/ 
For our system we did:

```
pip install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp36-cp36m-manylinux1_x86_64.whl 

pip install torchvision
```


# To activate the required conda environment

```
source activate mclv_env
```

# To run the program the following commands can be used
To display options 

```
python3 py/main.py -h
```

To Setup Data(Requires Tensorflow or email us for the processed dataset if you can't install tribeflow):
```
python3 py/setup_data.py -b `pwd`/ 
```


For PCD Results

```
python3 py/main.py -b `pwd`/ -n 1 --method PCD -cdk 10 -tot 100 --plateau 10 --hidden 32 --final-likelihood --filename h32pcd1_3
```

For CD Results

```
python3 py/main.py -b `pwd`/ -n 1 --method CD -cdk 10 -tot 100 --plateau 10 --hidden 32 --final-likelihood --filename h32pcd1_3
```

For MCLV Results

```
python3 py/main.py -b `pwd`/ -n 1 --method MCLV -mclvk 10 -cdk 10 -wm 15 -tot 100 --plateau 10 --hidden 32 --final-likelihood --filename h32pcd1_3
```


# Notes
You'll need a GPU with at least 8GB free memory, if not use the GPU-LIMIT to tune the parallelism for the likelihood computation.
We didn't implement bloom filters at the present but used python sets instead.