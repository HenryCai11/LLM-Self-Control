import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--batchsize", type=int, default=1)
    parser.add_argument("--prefix_length", type=int, default=10)
    parser.add_argument("--search", type=bool, default=True)
    parser.add_argument("--add_kl", type=bool, default=False)

    return parser.parse_args()

args = get_args()