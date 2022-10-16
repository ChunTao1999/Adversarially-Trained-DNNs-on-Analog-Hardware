g=2 #gpu card #
b1=512 #batch size, 2 for w4, 10 for w1
b2=128
l1=100 #limit test
l2=500
i1=1 #inflate
i2=4
#nl=0 #noise layer
a1='resnet10' #architecture
a2='resnet20'
c1='cifar10' #dataset
c2='cifar100'
d='./customdata/cifar100/Non-Adaptive-WB-attacks/' #custom directory
mvm1='32x32_100k' #mvm type
mvm2='64x64_100k'

#Pretrained Model Paths
pt0='./log/cifar10/clean-resnet10w1/' #pretrained directory 0
pt1='./log/cifar10/pgd-linf-eps2iter50-resnet10w1/' #pretrained directory 1
pt2='./log/cifar10/pgd-linf-eps4iter50-resnet10w1/' #pretrained directory 2
pt3='./log/cifar10/pgd-linf-eps6iter50-resnet10w1/' #pretrained directory 3
pt4='./log/cifar10/pgd-linf-eps8iter50-resnet10w1/' #pretrained directory 4
pt00='./log/cifar10/clean-resnet10w4/' #pretrained directory 0
pt11='./log/cifar10/pgd-linf-eps2iter50-resnet10w4/' #pretrained directory 1
pt22='./log/cifar10/pgd-linf-eps4iter50-resnet10w4/' #pretrained directory 2
pt33='./log/cifar10/pgd-linf-eps6iter50-resnet10w4/' #pretrained directory 3
pt44='./log/cifar10/pgd-linf-eps8iter50-resnet10w4/' #pretrained directory 4
pt000='./log/cifar100/clean-resnet20w1/' #pretrained directory 0
pt111='./log/cifar100/pgd-linf-eps2iter50-resnet20w1/' #pretrained directory 1
pt222='./log/cifar100/pgd-linf-eps4iter50-resnet20w1/' #pretrained directory 2
pt333='./log/cifar100/pgd-linf-eps6iter50-resnet20w1/' #pretrained directory 3
pt444='./log/cifar100/pgd-linf-eps8iter50-resnet20w1/' #pretrained directory 4
pt0000='./log/cifar100/clean-resnet20w4/' #pretrained directory 0
pt1111='./log/cifar100/pgd-linf-eps2iter50-resnet20w4/' #pretrained directory 1
pt2222='./log/cifar100/pgd-linf-eps4iter50-resnet20w4/' #pretrained directory 2
pt3333='./log/cifar100/pgd-linf-eps6iter50-resnet20w4/' #pretrained directory 3
pt4444='./log/cifar100/pgd-linf-eps8iter50-resnet20w4/' #pretrained directory 4

#Sigma Paths
pt_sigma0='./log/cifar10/clean-resnet10w1/clean/digital_vs_32x32_100k_findstd/dig_vs_mvm_std_N1000.npy'
pt_sigma1='./log/cifar10/pgd-linf-eps2iter50-resnet10w1/clean/digital_vs_32x32_100k_findstd/dig_vs_mvm_std_N1000.npy'
pt_sigma2='./log/cifar10/pgd-linf-eps4iter50-resnet10w1/clean/digital_vs_32x32_100k_findstd/dig_vs_mvm_std_N1000.npy'
pt_sigma3='./log/cifar10/pgd-linf-eps6iter50-resnet10w1/clean/digital_vs_32x32_100k_findstd/dig_vs_mvm_std_N1000.npy'
pt_sigma4='./log/cifar10/pgd-linf-eps8iter50-resnet10w1/clean/digital_vs_32x32_100k_findstd/dig_vs_mvm_std_N1000.npy'
pt_sigma00='./log/cifar10/clean-resnet10w4/clean/digital_vs_32x32_100k_findstd/dig_vs_mvm_std_N1000.npy'
pt_sigma11='./log/cifar10/pgd-linf-eps2iter50-resnet10w4/clean/digital_vs_32x32_100k_findstd/dig_vs_mvm_std_N1000.npy'
pt_sigma22='./log/cifar10/pgd-linf-eps4iter50-resnet10w4/clean/digital_vs_32x32_100k_findstd/dig_vs_mvm_std_N1000.npy'
pt_sigma33='./log/cifar10/pgd-linf-eps6iter50-resnet10w4/clean/digital_vs_32x32_100k_findstd/dig_vs_mvm_std_N1000.npy'
pt_sigma44='./log/cifar10/pgd-linf-eps8iter50-resnet10w4/clean/digital_vs_32x32_100k_findstd/dig_vs_mvm_std_N1000.npy'
pt_sigma000='./log/cifar100/clean-resnet20w1/clean/digital_vs_32x32_100k_findstd/dig_vs_mvm_std_N1000.npy'
pt_sigma111='./log/cifar100/pgd-linf-eps2iter50-resnet20w1/clean/digital_vs_32x32_100k_findstd/dig_vs_mvm_std_N1000.npy'
pt_sigma222='./log/cifar100/pgd-linf-eps4iter50-resnet20w1/clean/digital_vs_32x32_100k_findstd/dig_vs_mvm_std_N1000.npy'
pt_sigma333='./log/cifar100/pgd-linf-eps6iter50-resnet20w1/clean/digital_vs_32x32_100k_findstd/dig_vs_mvm_std_N1000.npy'
pt_sigma444='./log/cifar100/pgd-linf-eps8iter50-resnet20w1/clean/digital_vs_32x32_100k_findstd/dig_vs_mvm_std_N1000.npy'
pt_sigma0000='./log/cifar100/clean-resnet20w4/clean/digital_vs_32x32_100k_findstd/dig_vs_mvm_std_N1000.npy'
pt_sigma1111='./log/cifar100/pgd-linf-eps2iter50-resnet20w4/clean/digital_vs_32x32_100k_findstd/dig_vs_mvm_std_N1000.npy'
pt_sigma2222='./log/cifar100/pgd-linf-eps4iter50-resnet20w4/clean/digital_vs_32x32_100k_findstd/dig_vs_mvm_std_N1000.npy'
pt_sigma3333='./log/cifar100/pgd-linf-eps6iter50-resnet20w4/clean/digital_vs_32x32_100k_findstd/dig_vs_mvm_std_N1000.npy'
pt_sigma4444='./log/cifar100/pgd-linf-eps8iter50-resnet20w4/clean/digital_vs_32x32_100k_findstd/dig_vs_mvm_std_N1000.npy'

#Mean Paths
pt_mean0='./log/cifar10/clean-resnet10w1/clean/digital_vs_32x32_100k_findstd/dig_vs_mvm_mean_N1000.npy'
pt_mean1='./log/cifar10/pgd-linf-eps2iter50-resnet10w1/clean/digital_vs_32x32_100k_findstd/dig_vs_mvm_mean_N1000.npy'
pt_mean2='./log/cifar10/pgd-linf-eps4iter50-resnet10w1/clean/digital_vs_32x32_100k_findstd/dig_vs_mvm_mean_N1000.npy'
pt_mean3='./log/cifar10/pgd-linf-eps6iter50-resnet10w1/clean/digital_vs_32x32_100k_findstd/dig_vs_mvm_mean_N1000.npy'
pt_mean4='./log/cifar10/pgd-linf-eps8iter50-resnet10w1/clean/digital_vs_32x32_100k_findstd/dig_vs_mvm_mean_N1000.npy'
pt_mean00='./log/cifar10/clean-resnet10w4/clean/digital_vs_32x32_100k_findstd/dig_vs_mvm_mean_N1000.npy'
pt_mean11='./log/cifar10/pgd-linf-eps2iter50-resnet10w4/clean/digital_vs_32x32_100k_findstd/dig_vs_mvm_mean_N1000.npy'
pt_mean22='./log/cifar10/pgd-linf-eps4iter50-resnet10w4/clean/digital_vs_32x32_100k_findstd/dig_vs_mvm_mean_N1000.npy'
pt_mean33='./log/cifar10/pgd-linf-eps6iter50-resnet10w4/clean/digital_vs_32x32_100k_findstd/dig_vs_mvm_mean_N1000.npy'
pt_mean44='./log/cifar10/pgd-linf-eps8iter50-resnet10w4/clean/digital_vs_32x32_100k_findstd/dig_vs_mvm_mean_N1000.npy'
pt_mean000='./log/cifar100/clean-resnet20w1/clean/digital_vs_32x32_100k_findstd/dig_vs_mvm_mean_N1000.npy'
pt_mean111='./log/cifar100/pgd-linf-eps2iter50-resnet20w1/clean/digital_vs_32x32_100k_findstd/dig_vs_mvm_mean_N1000.npy'
pt_mean222='./log/cifar100/pgd-linf-eps4iter50-resnet20w1/clean/digital_vs_32x32_100k_findstd/dig_vs_mvm_mean_N1000.npy'
pt_mean333='./log/cifar100/pgd-linf-eps6iter50-resnet20w1/clean/digital_vs_32x32_100k_findstd/dig_vs_mvm_mean_N1000.npy'
pt_mean444='./log/cifar100/pgd-linf-eps8iter50-resnet20w1/clean/digital_vs_32x32_100k_findstd/dig_vs_mvm_mean_N1000.npy'
pt_mean0000='./log/cifar100/clean-resnet20w4/clean/digital_vs_32x32_100k_findstd/dig_vs_mvm_mean_N1000.npy'
pt_mean1111='./log/cifar100/pgd-linf-eps2iter50-resnet20w4/clean/digital_vs_32x32_100k_findstd/dig_vs_mvm_mean_N1000.npy'
pt_mean2222='./log/cifar100/pgd-linf-eps4iter50-resnet20w4/clean/digital_vs_32x32_100k_findstd/dig_vs_mvm_mean_N1000.npy'
pt_mean3333='./log/cifar100/pgd-linf-eps6iter50-resnet20w4/clean/digital_vs_32x32_100k_findstd/dig_vs_mvm_mean_N1000.npy'
pt_mean4444='./log/cifar100/pgd-linf-eps8iter50-resnet20w4/clean/digital_vs_32x32_100k_findstd/dig_vs_mvm_mean_N1000.npy'


mvm_stddev='32x32_100k'

eps1=10 #attack epsilon
eps2=12

#CUDA_VISIBLE_DEVICES=$g python evaluate_noise_injection_std+mean.py --dataset $c2 --arch $a2 --inflate $i2 --pretrained $pt0000 --batch-size $b2 --sigma-path $pt_sigma0000 --mean-path $pt_mean0000 --mvm-std $mvm_stddev
#CUDA_VISIBLE_DEVICES=$g python evaluate_noise_injection_std+mean.py --dataset $c2 --arch $a2 --inflate $i2 --pretrained $pt1111 --batch-size $b2 --sigma-path $pt_sigma1111 --mean-path $pt_mean1111 --mvm-std $mvm_stddev
#CUDA_VISIBLE_DEVICES=$g python evaluate_noise_injection_std+mean.py --dataset $c2 --arch $a2 --inflate $i2 --pretrained $pt2222 --batch-size $b2 --sigma-path $pt_sigma2222 --mean-path $pt_mean2222 --mvm-std $mvm_stddev
#CUDA_VISIBLE_DEVICES=$g python evaluate_noise_injection_std+mean.py --dataset $c2 --arch $a2 --inflate $i2 --pretrained $pt3333 --batch-size $b2 --sigma-path $pt_sigma3333 --mean-path $pt_mean3333 --mvm-std $mvm_stddev
#CUDA_VISIBLE_DEVICES=$g python evaluate_noise_injection_std+mean.py --dataset $c2 --arch $a2 --inflate $i2 --pretrained $pt4444 --batch-size $b2 --sigma-path $pt_sigma4444 --mean-path $pt_mean4444 --mvm-std $mvm_stddev

# First Eps
CUDA_VISIBLE_DEVICES=$g python evaluate_noise_injection_WB_attacks_std+mean.py --dataset $c1 --arch $a1 --inflate $i1 --pretrained $pt0 --attack-epsilon $eps1 --batch-size $b1 --sigma-path $pt_sigma0 --mean-path $pt_mean0 --mvm-std $mvm_stddev
CUDA_VISIBLE_DEVICES=$g python evaluate_noise_injection_WB_attacks_std+mean.py --dataset $c1 --arch $a1 --inflate $i1 --pretrained $pt1 --attack-epsilon $eps1 --batch-size $b1 --sigma-path $pt_sigma1 --mean-path $pt_mean1 --mvm-std $mvm_stddev
CUDA_VISIBLE_DEVICES=$g python evaluate_noise_injection_WB_attacks_std+mean.py --dataset $c1 --arch $a1 --inflate $i1 --pretrained $pt2 --attack-epsilon $eps1 --batch-size $b1 --sigma-path $pt_sigma2 --mean-path $pt_mean2 --mvm-std $mvm_stddev
CUDA_VISIBLE_DEVICES=$g python evaluate_noise_injection_WB_attacks_std+mean.py --dataset $c1 --arch $a1 --inflate $i1 --pretrained $pt3 --attack-epsilon $eps1 --batch-size $b1 --sigma-path $pt_sigma3 --mean-path $pt_mean3 --mvm-std $mvm_stddev
CUDA_VISIBLE_DEVICES=$g python evaluate_noise_injection_WB_attacks_std+mean.py --dataset $c1 --arch $a1 --inflate $i1 --pretrained $pt4 --attack-epsilon $eps1 --batch-size $b1 --sigma-path $pt_sigma4 --mean-path $pt_mean4 --mvm-std $mvm_stddev

CUDA_VISIBLE_DEVICES=$g python evaluate_noise_injection_WB_attacks_std+mean.py --dataset $c1 --arch $a1 --inflate $i2 --pretrained $pt00 --attack-epsilon $eps1 --batch-size $b2 --sigma-path $pt_sigma00 --mean-path $pt_mean00 --mvm-std $mvm_stddev
CUDA_VISIBLE_DEVICES=$g python evaluate_noise_injection_WB_attacks_std+mean.py --dataset $c1 --arch $a1 --inflate $i2 --pretrained $pt11 --attack-epsilon $eps1 --batch-size $b2 --sigma-path $pt_sigma11 --mean-path $pt_mean11 --mvm-std $mvm_stddev
CUDA_VISIBLE_DEVICES=$g python evaluate_noise_injection_WB_attacks_std+mean.py --dataset $c1 --arch $a1 --inflate $i2 --pretrained $pt22 --attack-epsilon $eps1 --batch-size $b2 --sigma-path $pt_sigma22 --mean-path $pt_mean22 --mvm-std $mvm_stddev
CUDA_VISIBLE_DEVICES=$g python evaluate_noise_injection_WB_attacks_std+mean.py --dataset $c1 --arch $a1 --inflate $i2 --pretrained $pt33 --attack-epsilon $eps1 --batch-size $b2 --sigma-path $pt_sigma33 --mean-path $pt_mean33 --mvm-std $mvm_stddev
CUDA_VISIBLE_DEVICES=$g python evaluate_noise_injection_WB_attacks_std+mean.py --dataset $c1 --arch $a1 --inflate $i2 --pretrained $pt44 --attack-epsilon $eps1 --batch-size $b2 --sigma-path $pt_sigma44 --mean-path $pt_mean44 --mvm-std $mvm_stddev

CUDA_VISIBLE_DEVICES=$g python evaluate_noise_injection_WB_attacks_std+mean.py --dataset $c2 --arch $a2 --inflate $i1 --pretrained $pt000 --attack-epsilon $eps1 --batch-size $b1 --sigma-path $pt_sigma000 --mean-path $pt_mean000 --mvm-std $mvm_stddev
CUDA_VISIBLE_DEVICES=$g python evaluate_noise_injection_WB_attacks_std+mean.py --dataset $c2 --arch $a2 --inflate $i1 --pretrained $pt111 --attack-epsilon $eps1 --batch-size $b1 --sigma-path $pt_sigma111 --mean-path $pt_mean111 --mvm-std $mvm_stddev
CUDA_VISIBLE_DEVICES=$g python evaluate_noise_injection_WB_attacks_std+mean.py --dataset $c2 --arch $a2 --inflate $i1 --pretrained $pt222 --attack-epsilon $eps1 --batch-size $b1 --sigma-path $pt_sigma222 --mean-path $pt_mean222 --mvm-std $mvm_stddev
CUDA_VISIBLE_DEVICES=$g python evaluate_noise_injection_WB_attacks_std+mean.py --dataset $c2 --arch $a2 --inflate $i1 --pretrained $pt333 --attack-epsilon $eps1 --batch-size $b1 --sigma-path $pt_sigma333 --mean-path $pt_mean333 --mvm-std $mvm_stddev
CUDA_VISIBLE_DEVICES=$g python evaluate_noise_injection_WB_attacks_std+mean.py --dataset $c2 --arch $a2 --inflate $i1 --pretrained $pt444 --attack-epsilon $eps1 --batch-size $b1 --sigma-path $pt_sigma444 --mean-path $pt_mean444 --mvm-std $mvm_stddev

CUDA_VISIBLE_DEVICES=$g python evaluate_noise_injection_WB_attacks_std+mean.py --dataset $c2 --arch $a2 --inflate $i2 --pretrained $pt0000 --attack-epsilon $eps1 --batch-size $b2 --sigma-path $pt_sigma0000 --mean-path $pt_mean0000 --mvm-std $mvm_stddev
CUDA_VISIBLE_DEVICES=$g python evaluate_noise_injection_WB_attacks_std+mean.py --dataset $c2 --arch $a2 --inflate $i2 --pretrained $pt1111 --attack-epsilon $eps1 --batch-size $b2 --sigma-path $pt_sigma1111 --mean-path $pt_mean1111 --mvm-std $mvm_stddev
CUDA_VISIBLE_DEVICES=$g python evaluate_noise_injection_WB_attacks_std+mean.py --dataset $c2 --arch $a2 --inflate $i2 --pretrained $pt2222 --attack-epsilon $eps1 --batch-size $b2 --sigma-path $pt_sigma2222 --mean-path $pt_mean2222 --mvm-std $mvm_stddev
CUDA_VISIBLE_DEVICES=$g python evaluate_noise_injection_WB_attacks_std+mean.py --dataset $c2 --arch $a2 --inflate $i2 --pretrained $pt3333 --attack-epsilon $eps1 --batch-size $b2 --sigma-path $pt_sigma3333 --mean-path $pt_mean3333 --mvm-std $mvm_stddev
CUDA_VISIBLE_DEVICES=$g python evaluate_noise_injection_WB_attacks_std+mean.py --dataset $c2 --arch $a2 --inflate $i2 --pretrained $pt4444 --attack-epsilon $eps1 --batch-size $b2 --sigma-path $pt_sigma4444 --mean-path $pt_mean4444 --mvm-std $mvm_stddev

# Second Eps
CUDA_VISIBLE_DEVICES=$g python evaluate_noise_injection_WB_attacks_std+mean.py --dataset $c1 --arch $a1 --inflate $i1 --pretrained $pt0 --attack-epsilon $eps2 --batch-size $b1 --sigma-path $pt_sigma0 --mean-path $pt_mean0 --mvm-std $mvm_stddev
CUDA_VISIBLE_DEVICES=$g python evaluate_noise_injection_WB_attacks_std+mean.py --dataset $c1 --arch $a1 --inflate $i1 --pretrained $pt1 --attack-epsilon $eps2 --batch-size $b1 --sigma-path $pt_sigma1 --mean-path $pt_mean1 --mvm-std $mvm_stddev
CUDA_VISIBLE_DEVICES=$g python evaluate_noise_injection_WB_attacks_std+mean.py --dataset $c1 --arch $a1 --inflate $i1 --pretrained $pt2 --attack-epsilon $eps2 --batch-size $b1 --sigma-path $pt_sigma2 --mean-path $pt_mean2 --mvm-std $mvm_stddev
CUDA_VISIBLE_DEVICES=$g python evaluate_noise_injection_WB_attacks_std+mean.py --dataset $c1 --arch $a1 --inflate $i1 --pretrained $pt3 --attack-epsilon $eps2 --batch-size $b1 --sigma-path $pt_sigma3 --mean-path $pt_mean3 --mvm-std $mvm_stddev
CUDA_VISIBLE_DEVICES=$g python evaluate_noise_injection_WB_attacks_std+mean.py --dataset $c1 --arch $a1 --inflate $i1 --pretrained $pt4 --attack-epsilon $eps2 --batch-size $b1 --sigma-path $pt_sigma4 --mean-path $pt_mean4 --mvm-std $mvm_stddev

CUDA_VISIBLE_DEVICES=$g python evaluate_noise_injection_WB_attacks_std+mean.py --dataset $c1 --arch $a1 --inflate $i2 --pretrained $pt00 --attack-epsilon $eps2 --batch-size $b2 --sigma-path $pt_sigma00 --mean-path $pt_mean00 --mvm-std $mvm_stddev
CUDA_VISIBLE_DEVICES=$g python evaluate_noise_injection_WB_attacks_std+mean.py --dataset $c1 --arch $a1 --inflate $i2 --pretrained $pt11 --attack-epsilon $eps2 --batch-size $b2 --sigma-path $pt_sigma11 --mean-path $pt_mean11 --mvm-std $mvm_stddev
CUDA_VISIBLE_DEVICES=$g python evaluate_noise_injection_WB_attacks_std+mean.py --dataset $c1 --arch $a1 --inflate $i2 --pretrained $pt22 --attack-epsilon $eps2 --batch-size $b2 --sigma-path $pt_sigma22 --mean-path $pt_mean22 --mvm-std $mvm_stddev
CUDA_VISIBLE_DEVICES=$g python evaluate_noise_injection_WB_attacks_std+mean.py --dataset $c1 --arch $a1 --inflate $i2 --pretrained $pt33 --attack-epsilon $eps2 --batch-size $b2 --sigma-path $pt_sigma33 --mean-path $pt_mean33 --mvm-std $mvm_stddev
CUDA_VISIBLE_DEVICES=$g python evaluate_noise_injection_WB_attacks_std+mean.py --dataset $c1 --arch $a1 --inflate $i2 --pretrained $pt44 --attack-epsilon $eps2 --batch-size $b2 --sigma-path $pt_sigma44 --mean-path $pt_mean44 --mvm-std $mvm_stddev

CUDA_VISIBLE_DEVICES=$g python evaluate_noise_injection_WB_attacks_std+mean.py --dataset $c2 --arch $a2 --inflate $i1 --pretrained $pt000 --attack-epsilon $eps2 --batch-size $b1 --sigma-path $pt_sigma000 --mean-path $pt_mean000 --mvm-std $mvm_stddev
CUDA_VISIBLE_DEVICES=$g python evaluate_noise_injection_WB_attacks_std+mean.py --dataset $c2 --arch $a2 --inflate $i1 --pretrained $pt111 --attack-epsilon $eps2 --batch-size $b1 --sigma-path $pt_sigma111 --mean-path $pt_mean111 --mvm-std $mvm_stddev
CUDA_VISIBLE_DEVICES=$g python evaluate_noise_injection_WB_attacks_std+mean.py --dataset $c2 --arch $a2 --inflate $i1 --pretrained $pt222 --attack-epsilon $eps2 --batch-size $b1 --sigma-path $pt_sigma222 --mean-path $pt_mean222 --mvm-std $mvm_stddev
CUDA_VISIBLE_DEVICES=$g python evaluate_noise_injection_WB_attacks_std+mean.py --dataset $c2 --arch $a2 --inflate $i1 --pretrained $pt333 --attack-epsilon $eps2 --batch-size $b1 --sigma-path $pt_sigma333 --mean-path $pt_mean333 --mvm-std $mvm_stddev
CUDA_VISIBLE_DEVICES=$g python evaluate_noise_injection_WB_attacks_std+mean.py --dataset $c2 --arch $a2 --inflate $i1 --pretrained $pt444 --attack-epsilon $eps2 --batch-size $b1 --sigma-path $pt_sigma444 --mean-path $pt_mean444 --mvm-std $mvm_stddev

CUDA_VISIBLE_DEVICES=$g python evaluate_noise_injection_WB_attacks_std+mean.py --dataset $c2 --arch $a2 --inflate $i2 --pretrained $pt0000 --attack-epsilon $eps2 --batch-size $b2 --sigma-path $pt_sigma0000 --mean-path $pt_mean0000 --mvm-std $mvm_stddev
CUDA_VISIBLE_DEVICES=$g python evaluate_noise_injection_WB_attacks_std+mean.py --dataset $c2 --arch $a2 --inflate $i2 --pretrained $pt1111 --attack-epsilon $eps2 --batch-size $b2 --sigma-path $pt_sigma1111 --mean-path $pt_mean1111 --mvm-std $mvm_stddev
CUDA_VISIBLE_DEVICES=$g python evaluate_noise_injection_WB_attacks_std+mean.py --dataset $c2 --arch $a2 --inflate $i2 --pretrained $pt2222 --attack-epsilon $eps2 --batch-size $b2 --sigma-path $pt_sigma2222 --mean-path $pt_mean2222 --mvm-std $mvm_stddev
CUDA_VISIBLE_DEVICES=$g python evaluate_noise_injection_WB_attacks_std+mean.py --dataset $c2 --arch $a2 --inflate $i2 --pretrained $pt3333 --attack-epsilon $eps2 --batch-size $b2 --sigma-path $pt_sigma3333 --mean-path $pt_mean3333 --mvm-std $mvm_stddev
CUDA_VISIBLE_DEVICES=$g python evaluate_noise_injection_WB_attacks_std+mean.py --dataset $c2 --arch $a2 --inflate $i2 --pretrained $pt4444 --attack-epsilon $eps2 --batch-size $b2 --sigma-path $pt_sigma4444 --mean-path $pt_mean4444 --mvm-std $mvm_stddev

