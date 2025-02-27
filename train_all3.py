import subprocess
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import json
from collections import OrderedDict
import datetime
import time



todo_combos_file = Path("queue_combos.json")
manage_combos_file = Path("manage_combos.json")

with todo_combos_file.open("r") as f:
    todo_combos = json.load(f)


if manage_combos_file.exists():
    with manage_combos_file.open("r") as f:
        manage_combos = json.load(f)
else:
    manage_combos = {
        "done": [],
        "todo": todo_combos,
        "doing": []
    }


# any doing combos that are not in todo_combos should be moved to todo_combos, in the beginning of list
if len(manage_combos["doing"]) > 0:
    manage_combos["todo"] = manage_combos["doing"] + manage_combos["todo"]
    manage_combos["doing"] = []



# Define the parameters
backbones = [
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

cls_loss_names = [
    "CELoss",
    "FocalLoss"
]

feature_loss_names = [
    "TripletLoss",
    # "HardMiningTripletLoss",
    # "SameDomainTripletLoss",
    "WeightedSoftTripletLoss",
    "ContrastiveLoss"
]

head = "SingleBranch"
cls_loss = "CELoss"
feature_loss = "WeightedSoftTripletLoss"
kl_loss = "KLLoss"

def is_combo_done(combo, checkpoint_dir):
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

def create_proc(*script_args):
    '''
        1train_test_local_py.sh accepts as arguments
            backbone=$1  # Accept backbone as an argument
            head=$2      # Accept head as an argument
            cls_loss=$3  # Accept cls_loss as an argument
            feature_loss=$4  # Accept feature_loss as an argument
            kl_loss=$5  # Accept kl_loss as an argument
    '''
    cmd = ["sh", "train_test_local_py.sh"] + list(script_args)
    # create file for output
    # timestamp string up to seconds
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    output_file = f"checkpoints/output_{script_args[0]}_{script_args[1]}_{timestamp}.log"
    f = open(output_file, "w")

    process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    return process, f


# Number of concurrent processes
concurrent_processes = 3



print(f"Total number of combinations: {len(todo_combos)}")

proc_active = []
comb_i = 0


default_args = OrderedDict(
    backbone = "ViTS-224",
    head = "SingleBranch",
    cls_loss = "CELoss",
    feature_loss = "WeightedSoftTripletLoss",
    kl_loss = "KLLoss",
)


while len(manage_combos["todo"]) >= 1:
    if len(proc_active) < concurrent_processes:
        combo = manage_combos["todo"].pop(0) # fetch new combo to process
        print(f"Starting process: {combo}")
        if is_combo_done(combo, "checkpoints"): # check if combo is already done
            print(f"\n\t\tCombo already done\n")
            print("-"*20)
            print("-"*20)
            continue

        manage_combos["doing"].append(combo) # add to doing list

        with manage_combos_file.open("w") as f: # update manage_combos file
            json.dump(manage_combos, f)

        args = default_args.copy() # create args for new combo
        args.update(combo)

        proc, f = create_proc(*list(args.values())) # create process for new combo
        proc_active.append((proc, f, combo)) # add to active processes



    # check if any active process is done
    new_done_inds = []
    for i, (proc, f, combo) in enumerate(proc_active):
        if proc.poll() is not None: # if process is done
            print(f"Process done: {combo}")
            f.close() # close the log file
            new_done_inds.append(i) # add to list of done processes, which will be removed from active processes

        manage_combos["doing"].remove(combo)
            

    # remove done processes from active processes
    # and add to done processes
    for i in new_done_inds[::-1]:
        proc_active.pop(i)

    time.sleep(0.5)


print("All done!")