1. Environments
	pytorch >= 1.8.0
	timm == 0.3.2
2. Data preparation
    2.1 You need to download ImageNet_LT, iNaturalist 2018, Places_LT, and then change the data_root in main_CNN.py and main_ViT.py accordingly.

    2.2After preparation, the file structures are as follows:
/path/to/ImageNet-LT/
    train/
        class1/
            img1.jpeg
        class2/
            img2.jpeg
    val/
        class1/
            img3.jpeg
        class2/
            img4.jpeg
    train.txt
    val.txt
    test.txt
    num_shots.txt
train.txt, val.txt and test.txt list the file names, and num_shots.txt gives the number of training images in each class. All these data files have been uploaded to this repo.

3. Usage
You can see all our settings in ./config/

Typically, 2 GPUs and >=24 GB per GPU Memory are available. But when training ViT-B-16 with a training resolution of 384, bigger GPU Memory is required.

For the stage one training, you can train the model with DataParallel or DistributedDataParallel. Specially, for stage one training, the commands are:

# Stage one
python main_CNN.py (or python main_ViT.py if you want to train ViT)
or
torch.distributed.launch --nproc_per_node=n main_CNN.py

where n is the number of gpus in your server. And you should divide the defaulting batch_size in our configs with n.

# Stage Two
python main_CNN_PC.py