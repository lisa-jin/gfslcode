
#Boosting Generalized Few-Shot Learning by Scattering Intra-Class Distribution
----
**Abstract**

Generalized Few-Shot Learning (GFSL) applies the model trained with the base classes to predict the samples from both base classes and novel classes, where each novel class is only provided with a few labeled samples during testing. 
Limited by the severe data imbalance between base and novel classes, GFSL easily suffers from the \textit{prediction shift issue} that most test samples tend to be classified into the base classes. Unlike the existing works that address this issue by either multi-stage training or complicated model design, we argue that extracting both discriminative and generalized feature representations is all GFSL needs, which could be achieved by simply scattering the intra-class distribution during training. Specifically, we introduce two self-supervised auxiliary tasks and a label permutation task to encourage the model to learn more image-level feature representations and push the decision boundary from novel towards base classes during inference. Our method is one-stage and could perform online inference. Experiments on the miniImageNet and tieredImageNet datasets show that the proposed method achieves comparable performance with the state-of-the-art multi-stage competitors under both traditional FSL and GFSL tasks, empirically proving that feature representation is the key for GFSL.

##Download data
miniImageNet and tieredImageNet can be downloaded from [fsl_data](https://www.dropbox.com/sh/6yd1ygtyc3yd981/AABVeEqzC08YQv4UZk7lNHvya?dl=0) and the auxiliary test images  for evaluating base classes are from [miniImageNet_aux](https://drive.google.com/drive/folders/1gk0AsYUrN9NtLIDE9AKaZeVYzftpo6Ol) and [tieredImageNet_aux](https://drive.google.com/drive/folders/19SGTIAnTmVohdSNN6Q9G5Qn3bVChOIMr).



##Running
We share Baseline and our method reproduction code for the article results.

###Baseline code
A reproduction of training code for miniImageNet:
```
CUDA_VISIBLE_DEVICES=0,1 python baseline_train.py --model resnet12 --method baseline --dataset 'miniImageNet' --model_path './save' --mode train --partion 1.0 --batch_size 64 --epochs 80 --lr_decay_epochs '60, 70' --data_root "./datasets/MiniImagenet/"
```
A reproduction of evaluating code for miniImageNet:
```
CUDA_VISIBLE_DEVICES=1,0 python baseline_val.py --model resnet12 --method baseline --dataset 'miniImageNet' --model_path './save' --mode val --test_state test  --partion 1.0 --task_mode gfsl --epochs 80 --batch_size 128 --data_root ./datasets/MiniImagenet/ --model_pretrained "./save/miniImageNet_baseline_resnet12/train/"
```
###Our method code
A reproduction of training code for miniImageNet:
```
CUDA_VISIBLE_DEVICES=0,1 python distill_gaussian.py --model resnet12 --method dis-gau_2 --dataset 'miniImageNet' --model_path './save' --alpha 1.0 --lamb_rot 1.0 --lamb_kl 0.0 --lamb_bk 0.0 --beta_rot 0.5 --beta_dis 0.5 --beta_3 0.5 --beta_4 0.5 --mode train --partion 1.0 --task_mode gfsl --test_state test --batch_size 64 --epochs 80 --lr_decay_epochs '60, 70' --data_root './datasets/MiniImagenet/'
```

##Acknowledge
We thank following repos providing helpful components/functions in our work.

 - [RFS](http://github.com/WangYueFt/rfs/)
 - [aCASTLE](https://github.com/Sha-Lab/aCASTLE)