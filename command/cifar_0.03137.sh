# eps_test = 2/255
python main.py \
--dataset CIFAR10 \
--model 'MLPModel(depth=6,width=5120,identity_val=10.0,scalar=True)' \
--loss 'mixture(lam0=0.05,lam_end=0.002)' \
--p-start 8 \
--p-end 1000 \
--epochs 0,0,100,1250,1300 \
--eps-test 0.00784 \
--eps-train 0.03922 \
-b 512 \
--lr 0.03 \
--scalar-lr 0.006 \
--gpu 0 \
-p 200 \
--seed 0 \
--visualize

# eps_test = 8/255 
python main.py \
--dataset CIFAR10 \
--model 'MLPModel(depth=6,width=5120,identity_val=10.0,scalar=True)' \
--loss 'mixture(lam0=0.1,lam_end=0.0005)' \
--p-start 8 \
--p-end 1000 \
--epochs 0,0,100,1250,1300 \
--eps-test 0.03137 \
--eps-train 0.09411 \
-b 512 \
--lr 0.03 \
--scalar-lr 0.006 \
--gpu 0 \
-p 200 \
--seed 0 \
--visualize

# eps_test = 16/255
python main.py \
--dataset CIFAR10 \
--model 'MLPModel(depth=6,width=5120,identity_val=10.0,scalar=True)' \
--loss 'mixture(lam0=0.1,lam_end=0.0002)' \
--p-start 8 \
--p-end 1000 \
--epochs 0,0,100,1250,1300 \
--eps-test 0.06274 \
--eps-train 0.09411 \
-b 512 \
--lr 0.03 \
--scalar-lr 0.006 \
--gpu 0 \
-p 200 \
--seed 0 \
--visualize