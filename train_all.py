import subprocess
from concurrent.futures import ThreadPoolExecutor
import os

print(os.getcwd())
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

def create_proc(*script_args):
    cmd = ["sh", "train_test_local_py.sh"] + list(script_args)
    # create file for output
    output_file = f"checkpoints/output_{script_args[0]}.log"
    f = open(output_file, "w")

    process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    return process, f


# Number of concurrent processes
concurrent_processes = 3

from itertools import product
import time

combinations = list(product(backbones, [head], [cls_loss], [feature_loss], [kl_loss]))
print(f"Total number of combinations: {len(combinations)}")

proc_active = []
proc_done = []
comb_i = 0

while len(proc_done) < len(combinations):
    if len(proc_active) < concurrent_processes and comb_i < len(combinations):
        proc, f = create_proc(*combinations[comb_i])
        proc_active.append((proc, f))
        comb_i += 1

    for i, (proc, f) in enumerate(proc_active):
        if proc.poll() is not None:
            f.close()
            proc_done.append(proc_active.pop(i))

    time.sleep(1)


print("All done!")