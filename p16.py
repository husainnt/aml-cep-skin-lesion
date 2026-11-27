import pandas as pd, numpy as np, os
import matplotlib.pyplot as plt

OUT = r"D:\ML CEP\CEP_CODE"
seeds = [42, 1, 2]
groups = {"image only (plain loss)": "effb0_plain_aug1_seed{}_history.csv",
          "hybrid (plain loss)": "hyb_effb0_plain_md0.0_seed{}_history.csv"}

os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
for ax, (name, pattern) in zip(axes, groups.items()):
    train, val = [], []
    for s in seeds:
        h = pd.read_csv(os.path.join(OUT, "outputs", pattern.format(s)))
        train.append(h["train_loss"].values)
        val.append(h["val_loss"].values)
        # here i print the epoch where the validation loss is lowest, and the losses at the end
        print(name, "seed", s, "| best val loss epoch:", int(h["val_loss"].idxmin()) + 1,
              "| min val loss %.3f" % h["val_loss"].min(),
              "| last epoch: train %.3f, val %.3f" % (h["train_loss"].iloc[-1], h["val_loss"].iloc[-1]))
    epochs = np.arange(1, len(train[0]) + 1)
    # here i draw the mean of the 3 seeds and a band from the lowest to the highest seed
    for values, label, color in [(train, "train loss", "tab:blue"), (val, "validation loss", "tab:orange")]:
        values = np.array(values)
        ax.plot(epochs, values.mean(axis=0), label=label, color=color)
        ax.fill_between(epochs, values.min(axis=0), values.max(axis=0), color=color, alpha=0.2)
    ax.set_title(name)
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)
    ax.legend()
axes[0].set_ylabel("cross-entropy loss")
plt.savefig(os.path.join(OUT, "figures", "fig_learning_curves.png"), dpi=200, bbox_inches="tight")
print("figure saved")