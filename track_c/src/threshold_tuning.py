import numpy as np
from sklearn.metrics import f1_score

def macro_f1(preds, labels):
    return f1_score(labels, preds, average="macro", labels=[0, 1, 2], zero_division=0.0)

def tune_thresholds_oof(oof_probs, true_labels, n_steps=100):
    """
    Mencari threshold terbaik per kelas untuk memaksimalkan Macro-F1 di OOF.
    Kita bergeser dari [0.0 ke 1.0] untuk masing-masing kelas.
    
    Pendekatan ini akan mengoptimalkan kelas Electronic (kelas 1) yang minoritas.
    """
    print("Memulai threshold tuning di OOF...")
    
    # Baseline dengan Argmax
    baseline_preds = np.argmax(oof_probs, axis=1)
    baseline_f1 = macro_f1(baseline_preds, true_labels)
    print(f"Baseline OOF Macro-F1 (Argmax): {baseline_f1:.5f}")
    
    best_thresholds = [1.0, 1.0, 1.0] # Pengali probabilitas dasar
    best_f1 = baseline_f1
    
    # Grid search sederhana: 
    # Karena kita ingin mem-boost probabilitas kelas tertentu,
    # kita kalikan probabilitas setiap kelas dengan weight, lalu ambil argmax.
    # W_0 = 1.0 (Fixed, sebagai jangkar)
    # W_1 = iterasi 0.5 s/d 2.0
    # W_2 = iterasi 0.5 s/d 2.0
    
    search_space = np.linspace(0.5, 2.0, num=30)
    
    for w1 in search_space:
        for w2 in search_space:
            weights = np.array([1.0, w1, w2])
            weighted_probs = oof_probs * weights
            preds = np.argmax(weighted_probs, axis=1)
            
            f1 = macro_f1(preds, true_labels)
            
            if f1 > best_f1:
                best_f1 = f1
                best_thresholds = [1.0, w1, w2]
                
    print(f"Tuning selesai")
    print(f"Best OOF Macro-F1: {best_f1:.5f} (Naik +{best_f1 - baseline_f1:.5f})")
    print(f"Best Multiplier: Kelas 0 = {best_thresholds[0]:.2f}, Kelas 1 = {best_thresholds[1]:.2f}, Kelas 2 = {best_thresholds[2]:.2f}")
    
    return best_thresholds

def apply_thresholds(test_probs, thresholds):
    """
    Menerapkan threshold hasil tuning ke probabilitas test.
    """
    weights = np.array(thresholds)
    weighted_probs = test_probs * weights
    return np.argmax(weighted_probs, axis=1)

if __name__ == "__main__":
    # Smoke test function
    pass
