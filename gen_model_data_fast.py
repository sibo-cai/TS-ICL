import os
import random
import numpy as np
import torch
from joblib import Parallel, delayed
import argparse


LENGTH = 1024

# time series generator
def generate_time_series_fast():
    t = np.linspace(0, 1, LENGTH)

    signal = np.zeros(LENGTH)

    # ---- 1. multiple frequency ----
    for _ in range(np.random.randint(1, 6)):
        freq = np.random.uniform(1, 50)
        phase = np.random.uniform(0, 2 * np.pi)
        amp = np.random.randn()
        signal += amp * np.sin(2 * np.pi * freq * t + phase)

    # ---- 2. trend----
    degree = np.random.randint(1, 4)
    coeffs = np.random.randn(degree + 1)
    trend = np.polyval(coeffs, t)

    # ---- 3. regime shift ----
    if np.random.rand() < 0.5:
        idx = np.random.randint(LENGTH // 4, 3 * LENGTH // 4)
        signal[idx:] += np.random.randn() * 2

    # ---- 4. spikes ----
    for _ in range(np.random.randint(0, 5)):
        pos = np.random.randint(0, LENGTH)
        signal[pos] += np.random.randn() * 3

    # ---- 5. noise ----
    noise_scale = np.random.rand(LENGTH) * 0.5
    noise = noise_scale * np.random.randn(LENGTH)

    # ---- 6. non linear transform ----
    if np.random.rand() < 0.3:
        signal = np.tanh(signal)

    return signal + trend + noise


# optional, pre-generated time series pool
def build_pool(pool_size=50000):
    return [generate_time_series_fast() for _ in range(pool_size)]


def sample_from_pool(pool):
    return pool[np.random.randint(len(pool))]


def generate_one_dataset(context_len, pool=None):
    # sampling target
    target_np = sample_from_pool(pool) if pool else generate_time_series_fast()
    target_tensor = torch.from_numpy(target_np).float()

    alpha = random.uniform(0, 0.5)
    ref_context_len = int(context_len * alpha)
    other_context_len = context_len - ref_context_len

    # ---- ref_context（repeating target）----
    if ref_context_len > 0:
        ref_context = target_tensor.unsqueeze(0).repeat(ref_context_len, 1)
    else:
        ref_context = None

    # sampling other_context
    if pool:
        other_context_np = np.stack([
            sample_from_pool(pool)
            for _ in range(other_context_len)
        ])
    else:
        other_context_np = np.stack([
            generate_time_series_fast()
            for _ in range(other_context_len)
        ])

    other_context = torch.from_numpy(other_context_np).float()

    # concatenation
    if ref_context_len > 0:
        context = torch.cat([ref_context, other_context], dim=0)
    else:
        context = other_context

    # ---- shuffling ----
    idx = torch.randperm(context.shape[0])
    context = context[idx]

    # ---- adding target ----
    sample = torch.cat([context, target_tensor.unsqueeze(0)], dim=0)

    return sample


def generate_model_data(args, context_len):
    print(f"Generating context_len={context_len}")

    pool = build_pool(args.pool_size) if args.use_pool else None

    datasets = Parallel(n_jobs=args.n_workers, backend="loky")(
        delayed(generate_one_dataset)(context_len, pool)
        for _ in range(args.n_samples)
    )

    datasets = torch.stack(datasets)

    save_file = os.path.join(
        args.output_dir,
        f"model_data_{context_len+1}_{context_len}_{args.flag}.npy"
    )

    # torch.save(datasets, save_file)
    np.save(save_file, datasets.numpy())
    print(f"Saved: {save_file}, shape={datasets.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./output_001')
    parser.add_argument('--context_len_min', type=int, default=30)
    parser.add_argument('--context_len_max', type=int, default=181)
    parser.add_argument('--context_len_interval', type=int, default=1)
    parser.add_argument('--n_samples', type=int, default=10000)
    parser.add_argument('--n_workers', type=int, default=2)
    parser.add_argument('--use_pool', action='store_true')
    parser.add_argument('--pool_size', type=int, default=50000)
    parser.add_argument('--flag', type=str, default='train')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    context_lens = list(range(
        args.context_len_min,
        args.context_len_max,
        args.context_len_interval
    ))

    for context_len in context_lens:
        generate_model_data(args, context_len)


if __name__ == "__main__":
    main()
