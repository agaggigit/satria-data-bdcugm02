"""
contact_sheet.py — Semi-Otomatis Review Visual
Track A — BDC Satria Data 2026

Membuat grid PNG (contact sheet) dari kandidat mislabel/ambigu.
Menggantikan inspeksi manual satu-per-satu: reviewer cukup lihat grid,
lalu putuskan per kelompok (keep/relabel/drop).

Fitur:
- Grid 5×10 = 50 gambar per halaman (bisa dikonfigurasi)
- Dikelompokkan per pasangan kelas (Recyclable→Organic, dsb.)
- Diurutkan dari paling mencurigakan (quality score / margin terendah)
- Caption per gambar: label asli, prediksi, margin, quality score
- Warna border: merah=mislabel yakin, oranye=ambigu, hijau=benar tapi margin rendah
- Output bisa ditampilkan langsung di Colab atau disimpan ke file

Jalankan via notebook 04_cleanlab_cleaning.ipynb di Colab.
"""

import math
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, UnidentifiedImageError


# ─── Konstanta ────────────────────────────────────────────────────────────────

CLASS_NAMES = ["Recyclable", "Electronic", "Organic"]
BORDER_COLORS = {
    "mislabel":  "#e74c3c",   # merah — label asli != prediksi (probable mislabel)
    "ambiguous": "#f39c12",   # oranye — label asli == prediksi tapi margin rendah
    "correct":   "#2ecc71",   # hijau — prediksi benar (sebagai referensi)
}


# ─── Helper ──────────────────────────────────────────────────────────────────

def _load_image(filepath: str, img_size: int = 224):
    """Load gambar, return PIL Image. Return placeholder jika gagal."""
    try:
        img = Image.open(filepath).convert("RGB")
        img = img.resize((img_size, img_size), Image.BILINEAR)
        return img
    except (FileNotFoundError, UnidentifiedImageError, Exception):
        # Placeholder abu-abu dengan teks error
        placeholder = Image.new("RGB", (img_size, img_size), color=(200, 200, 200))
        return placeholder


def _determine_border_color(row: pd.Series) -> str:
    """
    Tentukan warna border berdasarkan tipe kasus:
    - Merah: prediksi ≠ label asli (probable mislabel)
    - Oranye: prediksi == label asli tapi margin rendah (ambigu)
    - Hijau: prediksi benar (kontrol)
    """
    label = int(row.get("label", row.get("label_asli", -1)))
    pred  = int(row.get("pred_class", row.get("label_predicted", -1)))

    if label != pred:
        return BORDER_COLORS["mislabel"]
    elif row.get("margin", 1.0) < 0.3:
        return BORDER_COLORS["ambiguous"]
    else:
        return BORDER_COLORS["correct"]


def _make_caption(row: pd.Series, max_chars: int = 30) -> str:
    """Buat caption singkat untuk gambar."""
    label = int(row.get("label", row.get("label_asli", -1)))
    pred  = int(row.get("pred_class", row.get("label_predicted", -1)))
    margin = row.get("margin", float("nan"))
    score  = row.get("label_quality_score", float("nan"))

    label_name = CLASS_NAMES[label] if 0 <= label <= 2 else "?"
    pred_name  = CLASS_NAMES[pred]  if 0 <= pred  <= 2 else "?"

    if label == pred:
        caption = f"✓ {label_name}\nM={margin:.2f}"
    else:
        caption = f"✗ {label_name}→{pred_name}\nM={margin:.2f}"

    if not math.isnan(score):
        caption += f" Q={score:.2f}"

    return caption


# ─── Core: Generate Satu Halaman ─────────────────────────────────────────────

def _make_contact_sheet_page(
    rows: List[pd.Series],
    n_cols: int = 5,
    img_size: int = 160,
    title: str = "",
    figsize_per_img: float = 2.0,
) -> plt.Figure:
    """
    Buat satu halaman contact sheet dari list baris DataFrame.

    Args:
        rows          : list pd.Series (baris kandidat)
        n_cols        : jumlah kolom dalam grid
        img_size      : ukuran gambar dalam pixel sebelum display
        title         : judul halaman
        figsize_per_img: ukuran (inch) per sel gambar

    Returns:
        matplotlib Figure
    """
    n_imgs  = len(rows)
    n_rows  = math.ceil(n_imgs / n_cols)
    fig_w   = n_cols * figsize_per_img
    fig_h   = n_rows * (figsize_per_img + 0.6) + 0.5  # +0.6 untuk caption

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
    if n_rows == 1:
        axes = np.array([axes])
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for idx, row in enumerate(rows):
        r = idx // n_cols
        c = idx % n_cols
        ax = axes[r, c]

        # Load gambar
        img = _load_image(str(row.get("filepath", "")), img_size)
        ax.imshow(img)

        # Border berwarna
        border_color = _determine_border_color(row)
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)

        # Caption
        caption = _make_caption(row)
        ax.set_title(caption, fontsize=6.5, pad=2, wrap=True)
        ax.set_xticks([])
        ax.set_yticks([])

    # Sembunyikan axes kosong
    for idx in range(n_imgs, n_rows * n_cols):
        r = idx // n_cols
        c = idx % n_cols
        axes[r, c].axis("off")

    if title:
        fig.suptitle(title, fontsize=11, y=1.01, fontweight="bold")

    plt.tight_layout(pad=0.5, h_pad=0.8)
    return fig


# ─── Main: Generate Semua Halaman ────────────────────────────────────────────

def generate_contact_sheets(
    candidates: pd.DataFrame,
    output_dir: str,
    group_by_pair: bool = True,
    n_per_page: int = 50,
    n_cols: int = 10,
    img_size: int = 160,
    save_png: bool = True,
    display_in_notebook: bool = True,
    max_pages: Optional[int] = None,
) -> List[plt.Figure]:
    """
    Generate semua contact sheet dari DataFrame kandidat.

    Args:
        candidates         : DataFrame kandidat (output ambiguous_filter atau cleanlab_runner)
                             Harus punya kolom: filepath, label (atau label_asli),
                             pred_class (atau label_predicted), margin
        output_dir         : folder untuk simpan PNG
        group_by_pair      : kelompokkan per pasangan kelas (Recyclable→Organic, dsb.)
        n_per_page         : jumlah gambar per halaman (default 50)
        n_cols             : jumlah kolom (default 10, jadi 5 baris × 10 kolom)
        img_size           : resolusi display gambar
        save_png           : simpan ke file PNG
        display_in_notebook: tampilkan di Colab
        max_pages          : batasi jumlah halaman (None = semua)

    Returns:
        list matplotlib Figure
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Normalisasi nama kolom
    df = candidates.copy()
    if "label_asli" in df.columns:
        if "label" not in df.columns:
            df["label"] = df["label_asli"]
        else:
            df["label"] = df["label"].fillna(df["label_asli"])
            
    if "label_predicted" in df.columns:
        if "pred_class" not in df.columns:
            df["pred_class"] = df["label_predicted"]
        else:
            df["pred_class"] = df["pred_class"].fillna(df["label_predicted"])

    all_figs = []
    page_num = 0

    if group_by_pair:
        # Buat kolom pair key
        df["_pair_key"] = df.apply(
            lambda r: f"{CLASS_NAMES[int(r['label'])]} → {CLASS_NAMES[int(r['pred_class'])]}",
            axis=1,
        )

        # Prioritaskan Organic↔Recyclable
        priority_keys = [
            "Recyclable → Organic",
            "Organic → Recyclable",
        ]
        all_pairs = []
        for pk in priority_keys:
            if pk in df["_pair_key"].unique():
                all_pairs.append(pk)
        for pk in sorted(df["_pair_key"].unique()):
            if pk not in all_pairs:
                all_pairs.append(pk)

        for pair_key in all_pairs:
            pair_df = df[df["_pair_key"] == pair_key].reset_index(drop=True)
            n_pages_pair = math.ceil(len(pair_df) / n_per_page)

            print(f"\n[generate_contact_sheets] Pair: {pair_key} — "
                  f"{len(pair_df)} gambar → {n_pages_pair} halaman")

            for p in range(n_pages_pair):
                if max_pages and page_num >= max_pages:
                    print(f"  (dihentikan di {max_pages} halaman)")
                    return all_figs

                start = p * n_per_page
                end   = min(start + n_per_page, len(pair_df))
                page_rows = [pair_df.iloc[i] for i in range(start, end)]

                title = (f"Pair: {pair_key} | "
                         f"Hal. {p+1}/{n_pages_pair} | "
                         f"Total: {len(pair_df)} gambar")

                fig = _make_contact_sheet_page(page_rows, n_cols=n_cols, img_size=img_size, title=title)
                all_figs.append(fig)

                if save_png:
                    pair_slug = pair_key.replace(" ", "_").replace("→", "to")
                    fname = out_dir / f"contact_sheet_{pair_slug}_page{p+1:02d}.png"
                    fig.savefig(fname, dpi=100, bbox_inches="tight")
                    print(f"  Disimpan: {fname}")

                if display_in_notebook:
                    plt.show()
                else:
                    plt.close(fig)

                page_num += 1

    else:
        # Tanpa grouping — urut berdasarkan margin
        n_pages = math.ceil(len(df) / n_per_page)
        print(f"[generate_contact_sheets] {len(df)} gambar → {n_pages} halaman")

        for p in range(n_pages):
            if max_pages and p >= max_pages:
                break

            start = p * n_per_page
            end   = min(start + n_per_page, len(df))
            page_rows = [df.iloc[i] for i in range(start, end)]

            title = f"Kandidat Ambigu/Mislabel | Hal. {p+1}/{n_pages} | Total: {len(df)}"
            fig = _make_contact_sheet_page(page_rows, n_cols=n_cols, img_size=img_size, title=title)
            all_figs.append(fig)

            if save_png:
                fname = out_dir / f"contact_sheet_page{p+1:02d}.png"
                fig.savefig(fname, dpi=100, bbox_inches="tight")
                print(f"  Disimpan: {fname}")

            if display_in_notebook:
                plt.show()
            else:
                plt.close(fig)

    print(f"\n✅ {len(all_figs)} halaman contact sheet selesai!")
    return all_figs


# ─── Interactive: Rekam Keputusan ────────────────────────────────────────────

def init_cleaning_log(candidates: pd.DataFrame, output_path: str) -> pd.DataFrame:
    """
    Inisialisasi cleaning_log.csv dari candidates DataFrame.
    Semua keputusan di-set ke 'pending' dulu.

    Kolom output:
        filepath, label_asli, label_baru, keputusan, alasan, reviewer
    """
    df = candidates.copy()
    if "label" in df.columns:
        if "label_asli" not in df.columns:
            df["label_asli"] = df["label"]
        else:
            df["label_asli"] = df["label_asli"].fillna(df["label"])

    log = pd.DataFrame({
        "filepath":     df["filepath"].values,
        "fold":         df.get("fold", pd.Series([None]*len(df))).values,
        "label_asli":   df["label_asli"].values,
        "label_baru":   df["label_asli"].values,  # default: tidak berubah
        "keputusan":    "pending",   # akan diisi: keep / relabel / drop
        "alasan":       "",
        "reviewer":     "",
        "margin":       df.get("margin", pd.Series([float("nan")]*len(df))).values,
        "label_quality_score": df.get("label_quality_score", pd.Series([float("nan")]*len(df))).values,
        "is_double_flagged": df.get("is_double_flagged", pd.Series([False]*len(df))).values,
    })

    log.to_csv(output_path, index=False)
    print(f"[init_cleaning_log] Disimpan ke {output_path}")
    print(f"  {len(log):,} entri pending — isi kolom 'keputusan' dan 'alasan' setelah review")
    print("  Nilai valid untuk 'keputusan': keep / relabel / drop")

    return log


def validate_cleaning_log(log_path: str) -> pd.DataFrame:
    """
    Validasi cleaning_log.csv sebelum dipakai generate_v2.py.
    Pastikan semua keputusan sudah diisi dan valid.
    """
    log = pd.read_csv(log_path)

    valid_decisions = {"keep", "relabel", "drop"}
    pending = log[log["keputusan"] == "pending"]
    invalid = log[~log["keputusan"].isin(valid_decisions)]

    print(f"[validate_cleaning_log] Total entri  : {len(log):,}")
    print(f"  keep   : {(log['keputusan'] == 'keep').sum():,}")
    print(f"  relabel: {(log['keputusan'] == 'relabel').sum():,}")
    print(f"  drop   : {(log['keputusan'] == 'drop').sum():,}")
    print(f"  pending: {len(pending):,}")

    if len(pending) > 0:
        print(f"\n  ⚠️ {len(pending)} entri masih 'pending' — review belum selesai!")
    if len(invalid) > 0:
        print(f"\n  ⚠️ {len(invalid)} keputusan tidak valid: {invalid['keputusan'].unique()}")

    if len(pending) == 0 and len(invalid) == 0:
        print("\n  ✅ Log valid — siap untuk generate_v2.py")

    # Validasi: relabel harus punya label_baru yang berbeda
    relabel_rows = log[log["keputusan"] == "relabel"]
    bad_relabel = relabel_rows[relabel_rows["label_asli"] == relabel_rows["label_baru"]]
    if len(bad_relabel) > 0:
        print(f"  ⚠️ {len(bad_relabel)} entri 'relabel' tapi label_baru == label_asli")

    return log
