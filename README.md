# A PyTorch Implementation of DEMIPL

This is a PyTorch implementation of our paper "**Disambiguated Attention Embedding for Multi-Instance Partial-Label Learning**", (**NeurIPS'23**). 

Authors: Wei Tang, [Weijia Zhang](https://www.weijiazhangxh.com/), and [Min-Ling Zhang](http://palm.seu.edu.cn/zhangml/)

```
@inproceedings{tang2023demipl,
  author    = {Wei Tang and Weijia Zhang and Min{-}Ling Zhang},
  title     = {Disambiguated Attention Embedding for Multi-Instance Partial-Label Learning},
  booktitle = {Advances in Neural Information Processing Systems 36 (NeurIPS'23), New Orleans, LA},
  year      = {2023}
}
```



If you are interested in multi-instance partial-label learning, the seminal work [MIPLGP](http://palm.seu.edu.cn/zhangml/files/SCIS'23.pdf) may be helpful to you.

```
@article{tang2023mipl,
  author    = {Wei Tang and Weijia Zhang and Min{-}Ling Zhang},
  title     = {Multi-Instance Partial-Label Learning: Towards Exploiting Dual Inexact Supervision},
  journal   = {Science China Information Sciences},
  year      = {2023}
}
```



## Requirements

```sh
numpy==1.21.5
scikit_learn==1.3.1
scipy==1.7.3
torch==1.12.1+cu113
```

To install the requirement packages, please run the following command:

```sh
pip install -r requirements.txt
```



## Datasets

The datasets used in this paper can be found on this [link](http://palm.seu.edu.cn/zhangml/Resources.htm#MIPL_data).



## Demo

To reproduce the results of MNIST_MIPL dataset in the paper, please run the following command:

```sh
CUDA_VISIBLE_DEVICES=0 python main.py --ds MNIST_MIPL --ds_suffix 1 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.001
CUDA_VISIBLE_DEVICES=0 python main.py --ds MNIST_MIPL --ds_suffix 2 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.001
CUDA_VISIBLE_DEVICES=0 python main.py --ds MNIST_MIPL --ds_suffix 3 --lr 0.05 --epochs 100 --normalize false --w_entropy_A 0.001
```



This package is only free for academic usage. Have fun!
