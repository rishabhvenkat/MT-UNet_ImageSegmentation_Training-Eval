#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
import argparse
import os
from torch.utils.data import DataLoader
from dataset.dataset_Synapse import Synapsedataset
from model.MTUNet import MTUNet
from utils.test_Synapse import inference

# -------------------- ARGUMENTS --------------------
parser = argparse.ArgumentParser()

parser.add_argument("--checkpoint", required=True,
                    help="Path to trained model checkpoint")

parser.add_argument("--list_dir", default="./dataset/Synapse/lists_Synapse")
parser.add_argument("--root_dir", default="./dataset/Synapse")
parser.add_argument("--volume_path", default="./dataset/Synapse/test")
parser.add_argument("--z_spacing", default=10)

parser.add_argument("--num_classes", default=9)
parser.add_argument("--img_size", default=224)
parser.add_argument("--test_save_dir", default=os.path.abspath("./predictions"))

args = parser.parse_args()

args.test_save_dir = os.path.abspath(args.test_save_dir)
os.makedirs(args.test_save_dir, exist_ok=True)
print("Saving predictions to:", args.test_save_dir)


# -------------------- DEVICE --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------- MODEL --------------------
model = MTUNet(args.num_classes)

checkpoint = torch.load(args.checkpoint, map_location=device)
model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()

# Ensure prediction directory exists
if not os.path.exists(args.test_save_dir):
    os.makedirs(args.test_save_dir)

# Create predictions folder
os.makedirs(args.test_save_dir, exist_ok=True)

# -------------------- TEST DATA --------------------
db_test = Synapsedataset(
    base_dir=args.volume_path,
    list_dir=args.list_dir,
    split="test"
)

testloader = DataLoader(
    db_test,
    batch_size=1,
    shuffle=False
)

# -------------------- RUN INFERENCE --------------------
with torch.no_grad():
    avg_dcs, avg_hd = inference(
        args,
        model,
        testloader,
        args.test_save_dir
    )

print("\n========== FINAL RESULTS ==========")
print("Mean Dice Score :", avg_dcs)
print("Mean HD95       :", avg_hd)
print("====================================")
