import pandas as pd
import os

def validate_submission(submission_path: str, template_path: str = None) -> bool:
    """
    Validator format wajib sebelum submit.
    Mengecek:
    1. Jumlah baris (1458)
    2. Nama dan urutan kolom ('id', 'predicted')
    3. Urutan id file test
    4. Nilai 'predicted' harus 0, 1, atau 2
    5. Tidak boleh ada missing value
    """
    print(f"Memeriksa file: {submission_path}...")
    
    if not os.path.exists(submission_path):
        print(f"FAILED: File {submission_path} tidak ditemukan")
        return False
        
    if template_path is None:
        raise ValueError("template_path wajib diisi — path ke submission.csv template panitia")
        
    try:
        sub_df = pd.read_csv(submission_path)
        template_df = pd.read_csv(template_path)
    except Exception as e:
        print(f"FAILED: Gagal membaca file CSV: {e}")
        return False

    is_valid = True

    # 1. Cek jumlah baris
    if len(sub_df) != len(template_df):
        print(f"FAILED: Jumlah baris salah (Harapan: {len(template_df)}, Aktual: {len(sub_df)})")
        is_valid = False
    else:
        print(f"OK: Jumlah baris benar ({len(sub_df)})")

    # 2. Cek nama kolom
    expected_cols = ['id', 'predicted']
    actual_cols = list(sub_df.columns)
    if actual_cols != expected_cols:
        print(f"FAILED: Kolom salah (Harapan: {expected_cols}, Aktual: {actual_cols})")
        is_valid = False
    else:
        print(f"OK: Nama kolom benar")

    # 3. Cek urutan baris id
    if is_valid: # hanya cek jika kolom id ada dan panjang baris sama
        if not sub_df['id'].equals(template_df['id']):
            print(f"FAILED: Urutan ID file berantakan, harus sama persis dengan {template_path}")
            is_valid = False
        else:
            print(f"OK: Urutan ID baris benar")

    # 4. Cek missing values
    if sub_df.isnull().any().any():
        print(f"FAILED: Terdapat data kosong (NaN)")
        is_valid = False
    else:
        print(f"OK: Tidak ada data kosong")

    # 5. Cek validitas label (harus 0, 1, atau 2)
    if 'predicted' in sub_df.columns:
        invalid_labels = sub_df[~sub_df['predicted'].isin([0, 1, 2])]
        if len(invalid_labels) > 0:
            print(f"FAILED: Ada label prediksi di luar 0/1/2 (Ditemukan: {sub_df['predicted'].unique()})")
            is_valid = False
        else:
            print(f"OK: Label prediksi hanya berisi 0/1/2")

    if is_valid:
        print("\nHASIL: FORMAT VALID. Siap disubmit.")
    else:
        print("\nHASIL: FORMAT TIDAK VALID. Perbaiki bug sebelum submit.")
        
    return is_valid

if __name__ == "__main__":
    # Test dummy
    pass
