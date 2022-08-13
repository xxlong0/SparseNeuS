# SparseNeuS: Fast Generalizable Neural Surface Reconstruction from Sparse views [ECCV2022]
We present a novel neural surface reconstruction method, called SparseNeuS, which can generalize to new scenes and work well with
sparse images (as few as 2 or 3).

![](./docs/images/teaser.jpg)

## [Project Page](https://www.xxlong.site/SparseNeuS/) | [Paper](https://arxiv.org/pdf/2206.05737.pdf) 

## Setup

### Dependencies
- pytorch
- [torchsparse](https://github.com/mit-han-lab/torchsparse)
- opencv_python
- trimesh
- numpy
- pyhocon
- icecream
- tqdm
- scipy
- PyMCubes

### Dataset
- DTU Training dataset. Please download the preprocessed DTU dataset provided by [MVSNet](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view).
- DTU testing dataset. Please download the testing dataset provided by [IDR](https://github.com/lioryariv/idr).
- BlendedMVS dataset.

### Training 
Our training has two stages. First train the coarse level and then the fine level.
```shell
python exp_runner_generic.py --mode train --conf ./confs/general_lod0.conf
python exp_runner_generic.py --mode train --conf ./confs/general_lod0.conf
```

### Finetuning

### Evaluation


## Citation

Cite as below if you find this repository is helpful to your project:

```
@article{long2022sparseneus,
          title={SparseNeuS: Fast Generalizable Neural Surface Reconstruction from Sparse views},
          author={Long, Xiaoxiao and Lin, Cheng and Wang, Peng and Komura, Taku and Wang, Wenping},
          journal={ECCV},
          year={2022}
        }
```

## Acknowledgement

Some code snippets are borrowed from [IDR](https://github.com/lioryariv/idr), [NeuS](https://github.com/Totoro97/NeuS) and [IBRNet](https://github.com/googleinterns/IBRNet). Thanks for these great projects.