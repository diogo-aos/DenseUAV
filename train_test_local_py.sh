#name="baseline"
root_dir="/datasets/DenseUAV/"
data_dir=$root_dir/train
test_dir=$root_dir/test
gpu_ids=0
num_worker=8
lr=0.01
batchsize=16
sample_num=1
block=1
num_bottleneck=512
backbone=$1  # Accept backbone as an argument
head=$2      # Accept head as an argument
head_pool="avg" # global avg max avg+max
cls_loss=$3  # Accept cls_loss as an argument
feature_loss=$4  # Accept feature_loss as an argument
kl_loss=$5  # Accept kl_loss as an argument
h=224
w=224
load_from="no"
ra="satellite"  # random affine
re="satellite"  # random erasing
cj="no"  # color jitter
rr="uav"  # random rotate

echo "backbone = $backbone"
echo "head = $head"
echo "cls_loss = $cls_loss"
echo "feature_loss = $feature_loss"
echo "kl_loss = $kl_loss"

# Generate a timestamp in the format YYYYMMDD_HHMMSS
timestamp=$(date +"%Y%m%d_%H%M%S")

# Include the timestamp in the name variable
name="${backbone}_${head}_${timestamp}"

python train.py --name $name --data_dir $data_dir --gpu_ids $gpu_ids --sample_num $sample_num \
                --block $block --lr $lr --num_worker $num_worker --head $head  --head_pool $head_pool \
                --num_bottleneck $num_bottleneck --backbone $backbone --h $h --w $w --batchsize $batchsize --load_from $load_from \
                --ra $ra --re $re --cj $cj --rr $rr --cls_loss $cls_loss --feature_loss $feature_loss --kl_loss $kl_loss

cd checkpoints/$name
python test.py --name $name --test_dir $test_dir --gpu_ids $gpu_ids --num_worker $num_worker
python evaluate_gpu.py
python evaluateDistance.py --root_dir $root_dir
cd ../../
