# SoSN-wacv19-master

Implementation of IEEE WACV2019 "Power Normalizing Second-order Similarity Network for Few-shot Learning"

![Pipline of SoSN](docs/sosn.png)

__Requires.__
```
pytorch-0.4.1
numpy
scipy
```

__For miniImagenet training and testing, run following commands.__

```
cd ./miniimagenet
python miniimagenet_train_few_shot_SoSN.py -w 5 -s 1 -sigma 100
python miniimagenet_test_few_shot_SoSN.py -w 5 -s 1 -sigma 100
```

## Citation
@inproceedings{zhang2019power,
  title={Power normalizing second-order similarity network for few-shot learning},
  author={Zhang, Hongguang and Koniusz, Piotr},
  booktitle={2019 IEEE Winter Conference on Applications of Computer Vision (WACV)},
  pages={1185--1193},
  year={2019},
  organization={IEEE}
}
