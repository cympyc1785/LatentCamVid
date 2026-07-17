import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def attention_bar(
    tokens,
    attn,
    topk=None,
    save_path=None,   # e.g. "attention.png"
    dpi=200,
    show=False
):
    if hasattr(attn, "detach"):
        attn = attn.detach().cpu().numpy()

    if topk is not None:
        idx = np.argsort(attn)[-topk:][::-1]
        tokens = [tokens[i] for i in idx]
        attn = attn[idx]

    tokens = [f"{tok} [{i}]" for i, tok in enumerate(tokens)]

    plt.figure(figsize=(8, max(2, len(tokens) * 0.4)))
    plt.barh(tokens, attn)
    plt.gca().invert_yaxis()
    plt.xlabel("Attention weight")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved to {save_path}")

    if show:
        plt.show()

    plt.close()


def attn_to_color(attn, cmap="viridis"):
    """
    attn: (N,)
    return: (N, 3) RGB in [0,1]
    """
    attn = attn.astype(np.float32)
    attn_norm = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)

    colormap = plt.get_cmap(cmap)
    colors = colormap(attn_norm)[:, :3]  # drop alpha

    return colors