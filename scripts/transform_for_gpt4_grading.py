import os
import argparse
import jsonlines
import sys
import pandas as pd
import numpy as np

sys.path.append("./")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text using a Language model")
    parser.add_argument("--path_1", type=str, required=True, help="Path to the first triplets")
    parser.add_argument("--path_2", type=str, required=True, help="Path to the second triplets")
    parser.add_argument("--save_path", type=str, required=True, help="Path where to save the transformed data")
    parser.add_argument("--target_path", type=str, default="")
    return parser.parse_args()

def main():
    np.random.seed(0)
    args = parse_args()
    data = []
    df_1 = pd.read_csv(args.path_1)
    df_2 = pd.read_csv(args.path_2)
    if args.target_path == "":
        for row_1, row_2 in zip(df_1.iterrows(), df_2.iterrows()):
            row_1 = row_1[1]
            row_2 = row_2[1]
            assert(row_1["text"] == row_2["text"])
            sample = {"text": row_1["text"], "triplet_set_1": row_1["output_0"], "triplet_set_2": row_2["output_0"]}
            data.append(sample)
    else:
        target_data = []
        with jsonlines.open(args.target_path, 'r') as f:
            for line in f:
                target_data.append(line)
        for row_1, row_2, target_row in zip(df_1.iterrows(), df_2.iterrows(), target_data):
            row_1 = row_1[1]
            row_2 = row_2[1]
            assert(row_1["text"] == row_2["text"] == target_row["text"])
            random_triplets = np.random.choice([row_1["output_0"], row_2["output_0"]])
            sample = {"text": row_1["text"], "triplet_set_1": target_row["triplets"], "triplet_set_2": random_triplets}
            data.append(sample)

    with jsonlines.open(args.save_path, 'w') as f:
        for sample in data:
            f.write(sample)


if __name__ == "__main__":
    main()