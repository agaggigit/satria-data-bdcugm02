import json
import numpy as np
import pandas as pd
import torch
from config import CFG
from model import build_model
from losses_metrics import macro_f1
from seed_utils import set_seed


def collect_oof():
    set_seed(CFG.seed)
    device = "cuda"
    df = pd.read_csv(CFG.folds_csv)
    n = len(df)
    oof = np.full((n, 3), np.nan, dtype=np.float32)

    from dataset import get_loaders

    for fold in range(5):
        model = build_model(num_classes=CFG.num_classes, drop_path_rate=CFG.drop_path_rate)
        state = torch.load(f"{CFG.save_dir}/fold{fold}.pt", map_location=device, weights_only=True)
        model.load_state_dict(state)
        model = model.to(device).eval()

        _, val_loader = get_loaders(fold=fold, img_size=CFG.img_size, batch=CFG.batch)
        val_idx = df.index[df["fold"] == fold].to_numpy()

        probs_list = []
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(images)
                probs_list.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        probs = np.concatenate(probs_list, axis=0)

        assert len(probs) == len(val_idx), \
            f"❌ fold {fold}: prediksi {len(probs)} != val rows {len(val_idx)}"
        oof[val_idx] = probs
        print(f"✅ fold {fold}: {len(val_idx)} sampel terisi")
        del model
        torch.cuda.empty_cache()

    assert not np.isnan(oof).any(), "❌ Ada sampel tanpa prediksi OOF"
    labels = df["label"].to_numpy()
    preds = oof.argmax(axis=1)
    overall_f1 = macro_f1(preds, labels)

    np.save(f"{CFG.save_dir}/oof.npy", oof)
    meta = {
        "shape": list(oof.shape),
        "index_source": "folds.csv baris ke-i oof = baris ke-i folds.csv",
        "content": "softmax probabilities kelas 0/1/2",
        "oof_overall_macro_f1_argmax": float(overall_f1),
        "backbone": CFG.backbone,
        "img_size": CFG.img_size,
        "accum_steps": CFG.accum_steps,
        "seed": CFG.seed,
        "tta_hint_test_size": 288,
    }
    with open(f"{CFG.save_dir}/oof_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n📊 OOF overall Macro-F1 (argmax): {overall_f1:.4f}")
    print(f"✅ oof.npy + oof_meta.json tersimpan di {CFG.save_dir}")
    return overall_f1


if __name__ == "__main__":
    collect_oof()
