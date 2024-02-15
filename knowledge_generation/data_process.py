import os
import argparse
import random

from utils import (
    csv_to_jsonl,
    read_csv,
    write_array2tsv,
    head_based_split,
    tsv_to_csv,
    csv_to_bart,
    jsonl_to_json,
)


def load_data(args):
    random.seed(args.random_seed)

    data_file = os.path.join(args.data_folder, "data.csv")

    data = read_csv(data_file, delimiter=",", skip_header=True)

    (train, val, test) = head_based_split(
        data,
        val_size=args.val_size,
        test_size=args.test_size,
        head_size_threshold=args.head_size_threshold,
    )

    return train, val, test


def main():
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-folder",
        type=str,
        help="Path to folder containing the data",
        default=os.path.join("data", "raw", "13b"),
    )
    parser.add_argument("--val-size", type=int, default=20000, help="Val size")
    parser.add_argument("--test-size", type=int, default=20000, help="Test size")
    parser.add_argument(
        "--head-size-threshold",
        type=int,
        default=500,
        help="Maximum number of tuples a head is involved in, "
        "in order to be a candidate for the val/test set",
    )
    parser.add_argument("--random-seed", type=int, default=30, help="Random seed")
    args = parser.parse_args()

    # Load data
    (train, val, test) = load_data(args)

    # Write files
    data_types = ["train", "test", "val"]
    header = ["head", "relation", "tails"]
    folder = args.data_folder
    for data_format in ["tsv", "csv", "jsonl", "json", "BART-format"]:
        if not os.path.exists(os.path.join(folder, data_format)):
            os.makedirs(os.path.join(folder, data_format))
    for data_type in data_types:
        # Write tsv files
        write_array2tsv(
            os.path.join(folder, "tsv", data_type + ".tsv"), eval(data_type)
        )
        # Write csv files
        tsv_to_csv(
            os.path.join(folder, "tsv", f"{data_type}.tsv"),
            os.path.join(folder, "csv", f"{data_type}.csv"),
            header,
        )
        # Write jsonl files
        csv_to_jsonl(
            os.path.join(folder, "csv", f"{data_type}.csv"),
            os.path.join(folder, "jsonl", data_type + ".jsonl"),
        )
        # Write json files
        jsonl_to_json(
            os.path.join(folder, "jsonl", f"{data_type}.jsonl"),
            os.path.join(folder, "json", f"{data_type}.json"),
        )
        # Write BART input format files
        csv_to_bart(
            os.path.join(folder, "csv", f"{data_type}.csv"),
            os.path.join(folder, "BART-format"),
            data_type,
        )


if __name__ == "__main__":
    main()
