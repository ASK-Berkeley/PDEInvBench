import argparse
from huggingface_hub import snapshot_download

datasets = [
    "darcy-flow-241",
    "darcy-flow-421",
    "korteweg-de-vries-1d",
    "navier-stokes-forced-2d-2048",
    "navier-stokes-forced-2d",
    "navier-stokes-unforced-2d",
    "reaction-diffusion-2d-du-512",
    "reaction-diffusion-2d-du",
    "reaction-diffusion-2d-k-512",
    "reaction-diffusion-2d-k",
]

splits = [
    "*",
    "train",
    "validation",
    "test",
    "out_of_distribution",
    "out_of_distribution_extreme",
]


def main():
    parser = argparse.ArgumentParser(
        description="Download PDE Inverse Problem Benchmarking datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="darcy-flow-241",
        choices=datasets,
        help="Dataset to download",
    )
    parser.add_argument(
        "--split", type=str, default="*", choices=splits, help="Data split to download"
    )
    parser.add_argument(
        "--local-dir", type=str, default="", help="Local directory to save data"
    )

    args = parser.parse_args()

    data_bucket = "DabbyOWL/PDE_Inverse_Problem_Benchmarking"

    print(f"Downloading {args.dataset}/{args.split} to {args.local_dir}")

    snapshot_download(
        data_bucket,
        allow_patterns=[f"{args.dataset}/{args.split}/*"],
        local_dir=args.local_dir,
        repo_type="dataset",
    )


if __name__ == "__main__":
    main()
