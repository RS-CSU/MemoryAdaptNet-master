# MemoryAdaptNet-master:Unsupervised Domain Adaptation Semantic Segmentation of High Resolution Remote Sensing Imagery with Invariant Domain-level Context memory

Pytorch implementation of our method for cross-domain semantic segmentation of the high-resolution remote sensing imagery. 

Contact: Jingru Zhu (zhujingru@csu.edu.cn)

## Paper
[Unsupervised Domain Adaptation Semantic Segmentation of HRS Imagery with Invariant Domain-level Context memory](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9667523) <br />
Jingru Zhu, Ya Guo , Geng Sun, Lobo Yang,  Min Deng and Jie Chen, Member, IEEE, <br />
submit to IEEE Transactions on Geoscience and Remote Sensing.


## Example Results


## Quantitative Reuslts


## Installation
* Install PyTorch from http://pytorch.org with Python 3.6 and PyTorch 1.8.0

* Clone this repo
```
git clone https://github.com/RS-CSU/MemoryAdaptNet-master
cd MemoryAdaptNet-master
```
## Dataset
* Download the [Potsdam Dataset](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx/) as the source domain, and put it in the `dataset/Potsdam` folder

* Download the [Vaihingen Dataset](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx/) as the target domain, and put it in the `data/Vaihingen` folder

## Testing
* Download the pre-trained [p2v_best_m.pth](https://pan.baidu.com/s/1ijuk4vzy8RIjNt2h55K5zA?pwd=nvjo) and put it in the `snapshots` folder
* Download the pre-trained [p2v_best.pth](https://pan.baidu.com/s/1ijuk4vzy8RIjNt2h55K5zA?pwd=nvjo) and put it in the `snapshots` folder

* Test the model and results will be saved in the `results` folder

* The ablation models
* Download the pre-trained [p2v_IDMA_m.pth](https://pan.baidu.com/s/1ijuk4vzy8RIjNt2h55K5zA?pwd=nvjo) and put it in the `snapshots` folder
* Download the pre-trained [p2v_IDMA.pth](https://pan.baidu.com/s/1ijuk4vzy8RIjNt2h55K5zA?pwd=nvjo) and put it in the `snapshots` folder
* Download the pre-trained [p2v_FDA+IDMA_m.pth](https://pan.baidu.com/s/1ijuk4vzy8RIjNt2h55K5zA?pwd=nvjo) and put it in the `snapshots` folder
* Download the pre-trained [p2v_FDA+IDMA.pth](https://pan.baidu.com/s/1ijuk4vzy8RIjNt2h55K5zA?pwd=nvjo) and put it in the `snapshots` folder
```
python test_p2v_v3_1.py
```

## Training Examples
* Train the Potsdam(IR-R-G) to Vaihingen(IR-R-G) model

```
python train_p2v_v3_1.py
```

## Related Implementation and Dataset
* 

## Acknowledgment
This code is heavily borrowed from [Pytorch-AdaptSegNet](https://github.com/wasidennis/AdaptSegNet) and [DeepLabv3Plus-Pytorch](https://github.com/VainF/DeepLabV3Plus-Pytorch).

## Note
The model and code are available for non-commercial research purposes only.

* 07/2022: code released




