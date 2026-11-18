"""
CS24B057 - Ch Venkata Sai Pranav Tej
EE506L Pattern Recognition and Machine Learning (PRML 2026) - Assignment 4 Capstone
Unified Reproducible Entry Point (Auto-detects CUDA GPU vs. CPU)

Usage:
  1. Default (Auto-selects 320x320 GPU Pipeline if CUDA is available, else 5-Fold CPU Pipeline):
       python CS24B057.py
  2. Specify custom dataset folder (containing student_pack / train / val / test_unlabeled):
       python CS24B057.py --data-dir /path/to/dataset
  3. Force GPU or CPU pipeline explicitly:
       python CS24B057.py --mode gpu --img-size 320 --epochs 15 --batch-size 32
       python CS24B057.py --mode cpu
  4. Run the Classical PRML Ablation Suite (Logistic Regression, PCA+LR, PCA+SVM, HOG+PCA+SVM):
       python CS24B057.py --ablations
"""

import os
import sys
import argparse
import torch


def main():
    parser = argparse.ArgumentParser(description="CS24B057 PRML 2026 PA4 Unified Reproducible Runner")
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "gpu", "cpu"],
                        help="Execution pipeline: 'auto' (GPU if CUDA available, else CPU), 'gpu', or 'cpu'")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to dataset root directory")
    parser.add_argument("--img-size", type=int, default=320, help="Input image resolution for GPU ensemble (default: 320)")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size for GPU training (default: 32)")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs per GPU backbone (default: 15)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Base learning rate for GPU optimizer (default: 3e-4)")
    parser.add_argument("--retrain", action="store_true", help="Force retraining GPU backbones from scratch")
    parser.add_argument("--ablations", action="store_true", help="Also run the classical PRML ablation benchmark")
    args, unknown = parser.parse_known_args()

    root_dir = os.path.dirname(os.path.abspath(__file__))

    if args.ablations:
        print("\n[Step 0] Running Classical PRML Ablation Benchmark (Stages 1 to 4)...")
        sys.path.insert(0, root_dir)
        import run_ablations
        run_ablations.main()

    has_cuda = torch.cuda.is_available()
    has_gpu_cache = os.path.exists(os.path.join(root_dir, f"gpu_cache_densenet121_{args.img_size}.npz"))

    use_gpu = (args.mode == "gpu") or (args.mode == "auto" and (has_cuda or has_gpu_cache))

    if use_gpu:
        try:
            import torchvision  # noqa: F401
            sys.path.insert(0, os.path.join(root_dir, "gpu"))
            import CS24B057_gpu
            print("=" * 95)
            print("CS24B057 | Launching Winning 320x320 High-Resolution GPU Hybrid Pipeline")
            print("=" * 95)
            sys.argv = [
                sys.argv[0],
                "--img-size", str(args.img_size),
                "--batch-size", str(args.batch_size),
                "--epochs", str(args.epochs),
                "--lr", str(args.lr),
            ]
            if args.data_dir:
                sys.argv.extend(["--data-dir", args.data_dir])
            if args.retrain:
                sys.argv.append("--retrain")
            CS24B057_gpu.main()
            return
        except ImportError as e:
            if args.mode == "gpu":
                raise RuntimeError(f"GPU pipeline requires torchvision ({e}). Run: pip install -r requirements.txt") from e
            print(f"Notice: torchvision not found ({e}); falling back to standalone CPU pipeline.")

    print("=" * 95)
    print("CS24B057 | Launching Standalone 5-Fold Stratified CV CPU Pipeline")
    print("=" * 95)
    sys.path.insert(0, os.path.join(root_dir, "cpu"))
    import CS24B057_cpu
    CS24B057_cpu.main()


if __name__ == "__main__":
    main()
