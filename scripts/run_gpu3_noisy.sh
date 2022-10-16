g=3 #gpu card #
b1=10 #batch size, 2 for w4, 10 for w1
b2=2
l1=100 #limit test
l2=500
i1=1 #inflate
i2=4
nl=0 #noise layer
a1='resnet10' #architecture
a2='resnet20'
c1='cifar10' #dataset
c2='cifar100'
d='./customdata/cifar100/Non-Adaptive-WB-attacks/' #custom directory
mvm1='32x32_100k' #mvm type
mvm2='64x64_100k'
pt0='./log/cifar10/pgd-linf-eps6iter50-resnet10w1/' #pretrained directory 0
pt1='./log/cifar10/pgd-linf-eps6iter50-resnet10w4/' #pretrained directory 1
pt2='./log/cifar100/pgd-linf-eps6iter50-resnet20w1/' #pretrained directory 2
pt3='./log/cifar100/pgd-linf-eps6iter50-resnet20w4/' #pretrained directory 3
pt4='./log/cifar100/pgd-linf-eps8iter50-resnet20w4/' #pretrained directory 4
s=1 #10*relative sigma of GaussianNoise injection
eps=10 #attack epsilon

CUDA_VISIBLE_DEVICES=$g python compare_act_mvmstd_mean.py --dataset $c1 --arch $a1 --inflate $i1 --pretrained $pt0 --store-act $True --mvm $True --mvm-type $mvm2 --batch-size $b1 --limit-test $l1
CUDA_VISIBLE_DEVICES=$g python compare_act_mvmstd_mean.py --dataset $c1 --arch $a1 --inflate $i2 --pretrained $pt1 --store-act $True --mvm $True --mvm-type $mvm2 --batch-size $b2 --limit-test $l2
CUDA_VISIBLE_DEVICES=$g python compare_act_mvmstd_mean.py --dataset $c2 --arch $a2 --inflate $i1 --pretrained $pt2 --store-act $True --mvm $True --mvm-type $mvm2 --batch-size $b1 --limit-test $l1
CUDA_VISIBLE_DEVICES=$g python compare_act_mvmstd_mean.py --dataset $c2 --arch $a2 --inflate $i2 --pretrained $pt3 --store-act $True --mvm $True --mvm-type $mvm2 --batch-size $b2 --limit-test $l2
CUDA_VISIBLE_DEVICES=$g python compare_act_mvmstd_mean.py --dataset $c2 --arch $a2 --inflate $i2 --pretrained $pt4 --store-act $True --mvm $True --mvm-type $mvm2 --batch-size $b2 --limit-test $l2

#CUDA_VISIBLE_DEVICES=$g python evaluate_noise_injection.py --dataset $c --arch $a --inflate $i --pretrained $pt0 --batch-size $b 
#CUDA_VISIBLE_DEVICES=$g python evaluate_noise_injection.py --dataset $c --arch $a --inflate $i --pretrained $pt1 --batch-size $b 
#CUDA_VISIBLE_DEVICES=$g python evaluate_noise_injection.py --dataset $c --arch $a --inflate $i --pretrained $pt2 --batch-size $b 
#CUDA_VISIBLE_DEVICES=$g python evaluate_noise_injection.py --dataset $c --arch $a --inflate $i --pretrained $pt3 --batch-size $b 
#CUDA_VISIBLE_DEVICES=$g python evaluate_noise_injection.py --dataset $c --arch $a --inflate $i --pretrained $pt4 --batch-size $b 

#CUDA_VISIBLE_DEVICES=$g python evaluate_noise_injection_WB_attacks.py --dataset $c --arch $a --inflate $i --pretrained $pt0 --attack-epsilon $eps --batch-size $b 
#CUDA_VISIBLE_DEVICES=$g python evaluate_noise_injection_WB_attacks.py --dataset $c --arch $a --inflate $i --pretrained $pt1 --attack-epsilon $eps --batch-size $b 
#CUDA_VISIBLE_DEVICES=$g python evaluate_noise_injection_WB_attacks.py --dataset $c --arch $a --inflate $i --pretrained $pt2 --attack-epsilon $eps --batch-size $b 
#CUDA_VISIBLE_DEVICES=$g python evaluate_noise_injection_WB_attacks.py --dataset $c --arch $a --inflate $i --pretrained $pt3 --attack-epsilon $eps --batch-size $b 
#CUDA_VISIBLE_DEVICES=$g python evaluate_noise_injection_WB_attacks.py --dataset $c --arch $a --inflate $i --pretrained $pt4 --attack-epsilon $eps --batch-size $b 


