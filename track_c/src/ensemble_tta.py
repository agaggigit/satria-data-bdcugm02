import torch
import numpy as np
import sys
import os
import torchvision.transforms as T
from track_c.src.config_c import CFG_C
from track_c.src.inference import load_model, predict_test

def get_tta_transform(img_size: int = 224) -> T.Compose:
    """
    Transform TTA (Test-Time Augmentation).
    Selain standar eval, kita tambahkan Horizontal Flip untuk gambar yang sama.
    """
    return T.Compose([
        T.Resize(img_size),
        T.CenterCrop(img_size),
        T.RandomHorizontalFlip(p=1.0),  # Pasti di-flip
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def run_5fold_ensemble_inference(device='cuda'):
    """
    Menjalankan 5 model checkpoint secara bergantian dan merata-ratakan prediksinya.
    Jika config use_tta = True, maka akan merata-ratakan prediksi original dan TTA.
    """
    print("Memulai 5-Fold Ensemble Inference...")

    # Tambahkan project root ke sys.path agar bisa import track_a.src.dataset
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if project_root not in sys.path:
        sys.path.append(project_root)

    from track_a.src.dataset import make_test_loader

    # 1. Load test loader original
    # Nama parameter pakai 'submission_csv' sesuai signature dataset.py Track A
    test_loader_orig = make_test_loader(
        test_dir=CFG_C.test_dir,
        submission_csv=CFG_C.sample_sub_path,  # ← nama parameter yang benar
        img_size=CFG_C.img_size,
        batch_size=CFG_C.batch_size,
        num_workers=CFG_C.num_workers
    )
    
    n_test = len(test_loader_orig.dataset)
    ensemble_probs = np.zeros((n_test, 3), dtype=np.float32)
    
    for fold in range(5):
        print(f"\n--- Memproses Fold {fold} ---")
        checkpoint_path = f"{CFG_C.checkpoints_dir}/fold{fold}.pt"
        
        # Load Model
        model = load_model(
            checkpoint_path=checkpoint_path,
            backbone=CFG_C.backbone,
            num_classes=CFG_C.num_classes,
            drop_path_rate=CFG_C.drop_path_rate,
            device=device
        )
        
        # Prediksi Original
        print("Prediksi Original...")
        probs_orig = predict_test(model, test_loader_orig, device)
        
        probs_final_fold = probs_orig
        
        # Prediksi TTA jika diaktifkan
        if CFG_C.use_tta:
            print("Prediksi TTA (Horizontal Flip)...")
            # Kita modifikasi sementara transform test_loader_orig datasetnya
            original_transform = test_loader_orig.dataset.transform
            test_loader_orig.dataset.transform = get_tta_transform(CFG_C.img_size)
            
            probs_tta = predict_test(model, test_loader_orig, device)
            probs_final_fold = (probs_orig + probs_tta) / 2.0
            
            # Kembalikan transform
            test_loader_orig.dataset.transform = original_transform
            
        ensemble_probs += probs_final_fold
        
        # Bersihkan memori model sebelum next fold
        del model
        torch.cuda.empty_cache()
        
    # Rata-ratakan probabilitas dari ke-5 fold
    ensemble_probs /= 5.0
    
    print("\n✅ Ensemble Inference Selesai!")
    return ensemble_probs

if __name__ == "__main__":
    pass
