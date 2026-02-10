import re
import matplotlib.pyplot as plt
import argparse
import os
import pandas as pd

def parse_log(log_path):
    steps = []
    grad_norms = []
    losses = []
    
    with open(log_path, 'r') as f:
        for line in f:
            # Match standard log line
            # Epoch: [21/100] Step: [0/3104] Loss: 0.7877 ... Grad: 47.96 ...
            if "Step:" in line and "Grad:" in line:
                try:
                    step_match = re.search(r"Step: \[(\d+)/", line)
                    grad_match = re.search(r"Grad: ([\d\.]+)", line)
                    loss_match = re.search(r"Loss: ([\d\.]+)", line)
                    
                    if step_match and grad_match and loss_match:
                        steps.append(int(step_match.group(1)))
                        grad_norms.append(float(grad_match.group(1)))
                        losses.append(float(loss_match.group(1)))
                except:
                    continue
                    
    return steps, grad_norms, losses

def plot_gradients(log_path, output_dir):
    steps, grad_norms, losses = parse_log(log_path)
    
    if not steps:
        print("No gradient data found in log file.")
        return

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(grad_norms, label='Grad Norm', alpha=0.7)
    plt.axhline(y=1.0, color='r', linestyle='--', label='Clip Threshold (1.0)')
    plt.title("Gradient Norm History")
    plt.xlabel("Log Step")
    plt.ylabel("Norm")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(losses, label='Loss', color='orange', alpha=0.7)
    plt.title("Loss History")
    plt.xlabel("Log Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'grad_analysis.png')
    plt.savefig(save_path)
    print(f"Gradient analysis plot saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="checkpoint_512_5channel/train.log", help="Path to train.log")
    parser.add_argument("--out", default="visualization", help="Output directory")
    args = parser.parse_args()
    
    plot_gradients(args.log, args.out)
