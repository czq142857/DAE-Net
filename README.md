# DAE-Net
PyTorch implementation for paper DAE-Net: Deforming Auto-Encoder for fine-grained shape co-segmentation.

## Requirements
- Python 3 with numpy, h5py, scikit-image, opencv-python, and PyTorch.


## Datasets and weights
We use the data provided by [BAE-Net](https://github.com/czq142857/BAE-NET).
Please download the pre-processed ShapeNet Part dataset [here](https://github.com/czq142857/BAE-NET#datasets-and-weights).
To use your own data, please refer to [BAE-Net's data preparation code](https://github.com/czq142857/BAE-NET/tree/master/point_sampling).

We also provide the pre-trained network weights for ShapeNet Part dataset.
(The weights will be provided here in a personal cloud storage. Omitted for anonymity.)


## Training

To train a model on the chair category:
```
python main.py --train --sparse_loss_weight 0.002 --resolution 32 --branch_num 16 --data_dir ./bae_net_data/03001627_chair/ --data_file 03001627_vox --sample_dir samples/03001627_chair
```
where ```--sparse_loss_weight``` is a hyperparameter to control the granularity of the segmentation; ```--resolution``` indicates the voxel resolution of the training shapes; ```--branch_num``` controls the number of branches, i.e., the maximum number of parts; ```--data_dir``` and ```--data_file``` specify the training data; ```--sample_dir``` specifies the folder to save the intermediate results during training.

Due to randomness in the network initialization, the segmentation quality may vary for each run. If you do not get the desired segmentation, train the model again before changing the hyperparameters.
The hyperparameters used in the paper are provided in *train.sh*.


## Evaluation

Just replace ```--train``` with ```--iou``` to obtain the segmentation accuracy in mean per-part IOU. You might also want to replace the entire dataset ```03001627_vox``` with the testing set ```03001627_test_vox```. The script will test all checkpoints in the corresponding checkpoint folder and save the numbers there.
```
python main.py --iou --sparse_loss_weight 0.002 --resolution 32 --branch_num 16 --data_dir ./bae_net_data/03001627_chair/ --data_file 03001627_test_vox --sample_dir samples/03001627_chair
```


## Visualization

Replace ```--train``` with ```--test``` will give you the reconstructed shapes, as well as the ground truth shapes with both ground truth and predicted segmentation.

Replace ```--train``` with ```--template``` will give you the learned part templates.

Replace ```--train``` with ```--cluster``` will give you the shape clusters based on part existence scores. Use ```--sample_dir``` to specify a new folder to store all the shapes, and then you can visualize them using *render_and_get_mp4.py*. The script will produce a video showing a representative shape for each clustered group and the number of shapes the group contains.
```
python main.py --cluster --sparse_loss_weight 0.002 --resolution 32 --branch_num 16 --data_dir ./bae_net_data/03001627_chair/ --data_file 03001627_vox --sample_dir groups/03001627_chair
python render_and_get_mp4.py --sample_dir groups/03001627_chair
```