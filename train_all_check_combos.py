import subprocess
import os
from pathlib import Path
from itertools import product
import time



root_dir = Path("/datasets/DenseUAV/")

default_args = dict(
    name="baseline",

    root_dir=str(root_dir),
    data_dir=str(root_dir / "train"),
    test_dir=str(root_dir / "test"),
    gpu_ids="0",
    num_worker=8,
    lr=0.01,
    batchsize=16,
    sample_num=1,
    block=1,
    num_bottleneck=512,

    backbone="ViTS-224", # resnet50 ViTS-224 senet
    head="SingleBranch",
    head_pool="avg", # global avg max avg+max
    cls_loss="CELoss", # CELoss FocalLoss
    feature_loss="WeightedSoftTripletLoss", # TripletLoss HardMiningTripletLoss WeightedSoftTripletLoss ContrastiveLoss
    kl_loss="KLLoss", # KLLoss

    h=224,
    w=224,
    load_from="no",
    ra="satellite",  # random affine
    re="satellite",  # random erasing
    cj="no",  # color jitter
    rr="uav",  # random rotate

)


def dry_run_train(iteration, train_args_dict, timeout=30):
    # Define the directories and parameters
    name = train_args_dict.pop("name")

    root_dir = train_args_dict.pop("root_dir")
    data_dir = train_args_dict.pop("data_dir")
    test_dir = train_args_dict.pop("test_dir")
    gpu_ids = train_args_dict.pop("gpu_ids")
    num_worker = train_args_dict.pop("num_worker")
    lr = train_args_dict.pop("lr")
    batchsize = train_args_dict.pop("batchsize")
    sample_num = train_args_dict.pop("sample_num")
    block = train_args_dict.pop("block")
    num_bottleneck = train_args_dict.pop("num_bottleneck")
    
    backbone = train_args_dict.pop("backbone")
    head = train_args_dict.pop("head")
    head_pool = train_args_dict.pop("head_pool")
    cls_loss = train_args_dict.pop("cls_loss")
    feature_loss = train_args_dict.pop("feature_loss")
    kl_loss = train_args_dict.pop("kl_loss")
    
    h = train_args_dict.pop("h")
    w = train_args_dict.pop("w")
    load_from = train_args_dict.pop("load_from")
    ra = train_args_dict.pop("ra")
    re = train_args_dict.pop("re")
    cj = train_args_dict.pop("cj")
    rr = train_args_dict.pop("rr")

    # Define the name for the experiment
    name = f"train_dry_run_{iteration}"

    # Construct the training command
    train_command = (
        f"python train.py --name {name} --data_dir {data_dir} --gpu_ids {gpu_ids} "
        f"--sample_num {sample_num} --block {block} --lr {lr} --num_worker {num_worker} "
        f"--head {head} --head_pool {head_pool} --num_bottleneck {num_bottleneck} "
        f"--backbone {backbone} --h {h} --w {w} --batchsize {batchsize} --load_from {load_from} "
        f"--ra {ra} --re {re} --cj {cj} --rr {rr} --cls_loss {cls_loss} "
        f"--feature_loss {feature_loss} --kl_loss {kl_loss} --num_epochs=2"
    ).split()

    process = subprocess.Popen(train_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # for 30 seconds, check if process has failed every second
    # if it has return 1
    # otherwise kill after 30 seconds
    finished = False
    t_start = time.time()
    while time.time() - t_start < timeout:
        if process.poll() is not None:
            yield ('finished', time.time() - t_start)
            finished = True
            break
        else:
            yield ('running', time.time() - t_start)
    if not finished:
        process.kill()
        yield ('timeout', time.time() - t_start)





# Define the parameters
backbone_names = [
    "resnet50", "RKNet", "senet", "ViTS-224", "ViTS-384", "DeitS-224",
    "DeitB-224", "Pvtv2b2", "ViTB-224", "SwinB-224", "Swinv2S-256",
    "Swinv2T-256", "Convnext-T", "EfficientNet-B2", "EfficientNet-B3",
    "EfficientNet-B5", "EfficientNet-B6", "vgg16"
]

head_names = [
    "SingleBranch",
    "SingleBranchCNN",
    "SingleBranchSwin",
    "NetVLAD",
    "FSRA",
    "FSRA_CNN",
    "LPN",
    "LPN_CNN",
    "GeM"
]

head_pool_names = ["global", "avg", "max", "avg+max"]
head_pool_names = ["avg"]

cls_loss_names = [
    "CELoss",
    "FocalLoss"
]

feature_loss_names = [
    "TripletLoss",
    "WeightedSoftTripletLoss",
    "ContrastiveLoss"
]

import tqdm
import json




backbone_head_combos = list(product(backbone_names, head_names))

results = []


for it, (backbone, head) in tqdm.tqdm(enumerate(backbone_head_combos),
                                      total=len(backbone_head_combos)):
    print(f"Iteration: {it}/{len(backbone_head_combos)} | {backbone} | {head}")
    train_args_dict = default_args.copy()
    train_args_dict["backbone"] = backbone
    train_args_dict["head"] = head

    gen = dry_run_train(it, train_args_dict, timeout=10)
    for ret_code, time_elapsed in gen:

        if ret_code in ('finished', 'timeout'):
            results.append((backbone, head, ret_code, time_elapsed))
            print(f"\t ret_code: {ret_code}, time_elapsed: {time_elapsed}")
        time.sleep(0.25)

    with open("backbone_head_combo_results.json", "w") as f:
        json.dump(results, f)
    print("Done!")

results


valid_combos = [combo for combo in results if combo[2] == 'timeout']
len(valid_combos)


# which backbones have no timeout
backbones_with_valid  = set([combo[0] for combo in valid_combos])
all_backbones = set(backbone_names)
print(f"Backbones without valid combos: {all_backbones - backbones_with_valid}")



#
# CREATE ALL COMBOS
#


final_list = list(product(
    [combo[0:2] for combo in valid_combos],
    cls_loss_names, feature_loss_names))
len(final_list)


final_list_of_dicts = []
for (backbone, head), cls_loss, feature_loss in final_list:
    final_list_of_dicts.append({
        "backbone": backbone,
        "head": head,
        "cls_loss": cls_loss,
        "feature_loss": feature_loss
    })

with open("queue_combos.json", "w") as f:
    json.dump(final_list_of_dicts, f, indent=2)
    
























# remove directories that don't have a "pytorch_result_1.mat" file

checkpoint_dir = Path("checkpoints/")

# list all top level dirs in checkpoints
checkpoint_dirs = [f for f in checkpoint_dir.iterdir() if f.is_dir()]
checkpoint_dirs

valid_runs = [d for d in checkpoint_dirs if (d / "pytorch_result_1.mat").exists()]
valid_runs

invalid = [d for d in checkpoint_dirs if d not in valid_runs]
invalid

import shutil

for d in invalid:
    shutil.rmtree(d)

# 


# remove logs that failed
logs = [f for f in checkpoint_dir.iterdir() if f.is_file() and f.suffix == ".log"]
logs

# successful runs have somewhere "Training complete in"
successful_runs = [f for f in logs if b"Training complete in" in f.read_bytes()]
# remove all other logs
failed_runs = [f for f in logs if f not in successful_runs]
failed_runs

for f in failed_runs:
    f.unlink()
















# load params of all valid runs, in file opts.yaml
import yaml

params = []
for d in valid_runs:
    with open(d / "opts.yaml") as f:
        params.append(yaml.safe_load(f))
params


# load all combos
with open("queue_combos.json") as f:
    combos = json.load(f)

combos

queued_keys = list(combos[0].keys())

done_combos = [
    {k: combo[k] for k in queued_keys}
    for combo in params
]



def check_if_combo_done(combo, checkpoint_dir):
    import yaml
    checkpoint_dir = Path(checkpoint_dir) # parent dir

    # list all top level dirs in checkpoints
    checkpoint_dirs = [f for f in checkpoint_dir.iterdir() if f.is_dir()]
    checkpoint_dirs

    valid_runs = [d for d in checkpoint_dirs if (d / "pytorch_result_1.mat").exists()]
    valid_runs

    params = []
    for d in valid_runs:
        with open(d / "opts.yaml") as f:
            params.append(yaml.safe_load(f))

    done_combos = [
        {k: c[k] for k in list(combo.keys())}
        for c in params
    ]

    return combo in done_combos



