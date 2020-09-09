# StyleGAN 2 in PyTorch

Implementation of Analyzing and Improving the Image Quality of StyleGAN (https://arxiv.org/abs/1912.04958) in PyTorch with variation in discriminator and generator.

## Notice

My starting point is the implementation https://github.com/rosinality/stylegan2-pytorch but a lot of changes has been made for conditioning.

## Requirements

I have tested on:

* PyTorch 1.4.0
* CUDA 10.1

## Usage

### Datasets :
The class dataset will lead everything for training or generation by telling on what conditions we need to condition.


#### Initiating Dataset :
 When creating the dataset, we have to provide a path to the folder image path and a path to a csv with all the information of the image (ie also the condition) :
 > --folder FOLDER_PATH
 > --csv_path CSV_PATH
If the csv_path is not given, we assume that the csv_path is FOLDER_PATH+".csv".


We distinguish two type of dataset depending on the origin of the dataset. We have two origins :

- Stellar :
 > --dataset_type stellar
- Product Library on OVH:
 > --dataset_type unique 
 
The main difference is that a the Product library dataset only have one image for the different SKUs so the sap_id is enough to retrieve the image from the folder image. On the other end, the stellar dataset have multiple_views so the sap_id or image_id is not enough. We retrieve the name of the image in the folder through the akamai url with which we downloaded the dataset and that should be present in the stellar csv.

#### Conditionning :

When conditionning on categories, we have to give the name of the category (this category must be present in the columns of the csv) and which type of conditioning we want to apply:
- One hot vector conditioning :
 > --labels "sap_sub_function" "sap_aesthetic_style"
- Inspirationnal conditioning (ie randomly generated weights for creativity):
 > --labels_inspirationnal "sap_sub_function" "sap_aesthetic_style"
- Both :
 >  --labels "sap_sub_function" --labels_inspirationnal  "sap_aesthetic_style"

IMPORTANT : Multiple conditionning is only available with the Design Discriminator option. AM GAN and Bilinear must have one label category or one inspirationnal label category.

#### Mask-Conditionning :

When conditionning on mask, we add the option as the following :
> --mask

We sample from a folder whose path is FOLDER_PATH + _mask. The name of the mask of an image should be the exact same name of the original image inside the mask folder.


### Networks: 
#### Discriminator type :
We can choose among three discriminators that will change both the architectures and the loss of discriminator. Only the design discriminator allow multiple conditionning :
Example :
- bilinear (gets his name from the Temporal evolution with bilinear layer paper):
 > --discriminator_type bilinear
- AM-GAN (gets his name from the AMGAN paper) :
 > --discriminator_type AMGAN
- design :
 > --discriminator_type design


### Training :
train.py supports Weights & Biases logging. If you want to use it, add --wandb arguments to the script.

#### Training in a progressive manner :
We leave the possibility to train GAN in a progressive manner, ie the scale of the output augments at different iterations.
To do so, one should give the option --progressive when training, should give an original size, a max_size (default = 256 pixels), the number of iteration before the first upscale, a multiplier of the number of iteration for the following upscale.
Example :
 > --size 8 --progressive --upscale_every 2000 --upscale_factor 2 --max_size 256 
With the previous example, we will train for 2000 iterations with a size 8, then 4000 iterations with size 16, then 8000 iterations with size 32 and so on until size 256 is reached.

#### Controlling training :
We are given control to the different loss, regularisation with lambda_CONTROL_WANTED.
We can also add augmentation to the training dataset using :
 > --augment --augment_p AUGMENT_PROBABILITY 

#### Train in distributed setting :
You can train model in distributed setting by using the distributed API from torch :

> python -m torch.distributed.launch --nproc_per_node=N_GPU --master_port=PORT train.py --OPTION



### Generate samples

> python generate.py --dataset_type DATASET_TYPE --csv_path CSV_PATH --folder FOLDER_PATH --labels LABEL_CONDITIONNING --labels_inspirationnal LABEL_INSPIRATION --sample N_ELEMENT --pics N_PICS --ckpt PATH_CHECKPOINT  

Generate N_PICS with N_ELEMENT on each from checkpoint PATH_CHECKPOINT using the repartition of labels and inspirationnal labels created with the dataset option.

You should change your size (--size 256 for example) if you train with another dimension.   



## License

Model details and custom CUDA kernel codes are from official repostiories: https://github.com/NVlabs/stylegan2

Codes for Learned Perceptual Image Patch Similarity, LPIPS came from https://github.com/richzhang/PerceptualSimilarity

To match FID scores more closely to tensorflow official implementations, I have used FID Inception V3 implementations in https://github.com/mseitzer/pytorch-fid
