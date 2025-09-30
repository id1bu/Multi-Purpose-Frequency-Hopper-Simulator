
#!/usr/bin/env python3
"""
Frequency Hopping Simulator (Lightweight)
Author: Divyanshu Shukla
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def generate_hopping_sequence(num_slots, channels, mode="prn", avoid=None, seed=42):
    rng = np.random.default_rng(seed)
    n = len(channels)
    seq = []
    last = None
    for t in range(num_slots):
        if mode == "sequential":
            seq.append(channels[t % n])
        else:  # prn
            attempts = 0
            while True:
                idx = rng.integers(0, n)
                candidate = channels[idx]
                attempts += 1
                if avoid is None or not any(abs(candidate - a) < (channels[1]-channels[0])/2 for a in avoid):
                    if last is None or candidate != last:
                        seq.append(candidate)
                        last = candidate
                        break
                if attempts > 50:
                    seq.append(candidate)
                    last = candidate
                    break
    return np.array(seq)

def main():
    parser = argparse.ArgumentParser(description="Frequency Hopping Simulator")
    parser.add_argument("--num_time_slots", type=int, default=200)
    parser.add_argument("--num_channels", type=int, default=24)
    parser.add_argument("--band_min", type=float, default=1000.0)
    parser.add_argument("--band_max", type=float, default=2000.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dwell_time_ms", type=int, default=50)
    parser.add_argument("--avoid_list", nargs="*", type=float, default=[])
    parser.add_argument("--mode", choices=["prn", "sequential"], default="prn")
    parser.add_argument("--out_dir", default="output")
    args = parser.parse_args()

    channels = np.linspace(args.band_min, args.band_max, args.num_channels)
    hops = generate_hopping_sequence(args.num_time_slots, channels, mode=args.mode, avoid=args.avoid_list, seed=args.seed)

    time_ms = np.arange(args.num_time_slots) * args.dwell_time_ms
    channel_idx = np.array([np.argmin(np.abs(channels - f)) for f in hops])
    df = pd.DataFrame({
        "time_ms": time_ms,
        "freq_MHz": hops,
        "channel_idx": channel_idx,
        "dwell_time_ms": args.dwell_time_ms
    })

    os.makedirs(args.out_dir, exist_ok=True)
    df.to_csv(os.path.join(args.out_dir, "hops.csv"), index=False)

    # Frequency vs Time plot
    plt.figure(figsize=(12, 4))
    plt.scatter(df["time_ms"], df["freq_MHz"], s=18)
    plt.xlabel("Time (ms)")
    plt.ylabel("Frequency (MHz)")
    plt.title("Frequency Hopping: Frequency vs Time")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "fh_simulator.png"))
    plt.close()

    # Occupancy heatmap
    occupancy = np.zeros((args.num_time_slots, args.num_channels), dtype=int)
    for i, ch in enumerate(channel_idx):
        occupancy[i, ch] = 1

    plt.figure(figsize=(12, 4))
    plt.imshow(occupancy.T, aspect='auto', origin='lower')
    plt.xlabel("Time slot index")
    plt.ylabel("Channel index (low->high freq)")
    plt.title("Frequency Occupancy Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "fh_heatmap.png"))
    plt.close()

    summary = df.groupby("channel_idx").size().rename("hits").reset_index()
    summary["channel_freq_MHz"] = summary["channel_idx"].apply(lambda i: channels[int(i)])
    summary = summary.sort_values("channel_idx")
    summary.to_csv(os.path.join(args.out_dir, "hops_summary.csv"), index=False)

if __name__ == "__main__":
    main()
