"""curves.py — Plot kurva train/val + LR vs step (verifikasi warmup+cosine)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_curves(history_df, out_path: str) -> None:
    """history_df kolom: epoch, train_loss, val_loss, train_f1, val_f1."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="train_loss")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="val_loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(history_df["epoch"], history_df["train_f1"], label="train_macro_f1")
    axes[1].plot(history_df["epoch"], history_df["val_f1"], label="val_macro_f1")
    axes[1].set_title("Macro-F1")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_lr(lrs, out_path: str) -> None:
    """Bukti visual warmup + cosine benar: naik bertahap -> puncak -> turun."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(len(lrs)), lrs)
    ax.set_xlabel("step")
    ax.set_ylabel("lr")
    ax.set_title("LR schedule (warmup + cosine)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
