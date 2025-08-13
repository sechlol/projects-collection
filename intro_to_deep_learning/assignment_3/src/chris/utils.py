import json
import os
import sys
from datetime import datetime
from typing import Dict, Any

import pandas as pd

OUT_DIR = f"../results/"


def ensure_path_exists(path: str):
    # Ensures that all the folders to the given path exists, and if not creates them
    abs_path = os.path.abspath(path)
    os.makedirs(abs_path, exist_ok=True)
    return str(abs_path)


def read_cmd_args(defaults: Dict = None) -> Dict[str, Any]:
    args = defaults or {}

    for i in range(1, len(sys.argv), 2):
        # check if the argument has a corresponding value
        if i + 1 < len(sys.argv):
            # add the argument and its value to the dictionary
            try:
                v = sys.argv[i + 1]
                v = float(v) if "." in v else int(sys.argv[i + 1])
            except ValueError:
                v = sys.argv[i + 1]
            args[sys.argv[i]] = v
    return args


def _save_args(cmd_args, out_path):
    file_args = out_path + "args.json"
    with open(file_args, "w") as f:
        json.dump(cmd_args, f)


def save_results(history, test_acc, test_loss, cmd_args):
    folder_name = f"{test_acc*100:.0f}_{history[-1][1]*100:.0f}_{history[-1][3]*100:.0f}_{datetime.now():%m_%d_%H%M%S}/"
    out_path = OUT_DIR + folder_name
    ensure_path_exists(out_path)
    _save_history(history, out_path)
    _save_test_result(test_acc, test_loss, out_path)
    _save_args(cmd_args, out_path)
    print("Output dir:", out_path)


def _save_history(history, out_path: str):
    file_history = out_path + "train_history.csv"
    df = pd.DataFrame(history, columns=["epoch", "train_accuracy", "train_loss", "val_accuracy", "val_loss"])
    df.to_csv(file_history, index=False)
    print(f"\nTrain history data saved as", file_history)


def _save_test_result(test_acc, test_loss, out_path):
    file_test_result = out_path + "test_result.csv"
    with open(file_test_result, "w") as file:
        file.write("accuracy,loss\n")
        file.write(",".join(map(str, [test_acc, test_loss])))
        file.write("\n")
    print("\nTest Results saved as", file_test_result)


# Utility
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
