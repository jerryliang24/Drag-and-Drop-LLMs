import os
from calculate_acc import check_validity

valid_total = 0.0
acc_total = 0.0
count = 0
base_dir = "../results/common_sense_reasoning/xxx"

ckpts = ["xxxyyT"]
for ckpt in ckpts:
    for file in [f for f in os.listdir(base_dir) if f.startswith(ckpt)]:
        filepath = os.path.join(base_dir, file)

        if os.path.isfile(filepath):
            valid_, acc_ = check_validity(filepath)
            valid_total += valid_
            acc_total += acc_
            count += 1

    if count > 0:
        print(f"ckpt{ckpt}'s average acc: {acc_total / count:.4f}"
