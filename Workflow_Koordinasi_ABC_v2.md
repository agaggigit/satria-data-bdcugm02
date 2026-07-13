# Workflow v2 — Koordinasi A / B / C (Pasca-Review, 12–30 Juli)

**Sederhana:** 5 aturan, 5 gate, 1 peta paralel. Kalau bingung "boleh mulai belum?" → lihat tabel Gate. Kalau bingung "sambil nunggu ngapain?" → lihat kolom Kerja Paralel.

> ⚠️ **Status penting:** Submission 1 (safety net) BELUM terkirim. Ini bukan milik Track C sendirian — ini deadline tim. Target: **terkirim ≤ 16 Juli**, dari ensemble baseline v1 apa adanya. Tujuannya membuktikan pipeline unggah jalan ujung-ke-ujung, bukan mengejar skor.

---

## 5 Aturan (semua orang hafal ini)

1. **Konstanta tidak berubah:** mapping `0=Recyclable, 1=Electronic, 2=Organic` · seed `42` · semua orang pakai `folds_v2.csv` yang sama setelah rilis. Assert di tiap kode.
2. **Versi, bukan timpa.** File v1 tidak pernah di-overwrite (`folds.csv` vs `folds_v2.csv`, `fold0.pt` vs `convnext_fold0_v2.pt`). Owner-only write tetap berlaku.
3. **Perubahan konvensi = umumkan dulu.** `img_size` 224→288 kena aturan ini: B umumkan → C konfirmasi transform eval ikut → baru dipakai. Tidak ada perubahan senyap.
4. **Test hanya untuk prediksi akhir.** Semua tuning (threshold, bobot, TTA, cleaning) diputuskan di OOF/train. Melanggar = risiko diskualifikasi.
5. **Validator + manifest sebelum SETIAP unggahan.** Tanpa kecuali. Unggah H-1 dari deadline internal, bukan menit akhir (tie-break: yang duluan menang).

---

## 5 Gate (kapan track boleh mulai)

| Gate | Dari → Ke | Isi | Target | Yang boleh mulai setelahnya |
|------|-----------|-----|--------|------------------------------|
| **G1** | B → A, C | `oof.npy` v1 lengkap + 5 checkpoint v1 | 13–14 Juli | A: cleaning sungguhan · C: uji modul di OOF asli + **rakit safety net** |
| **G-SUB1** | C → panitia | Submission 1 (safety net), validator lolos | **≤ 16 Juli** | Tim tenang; pipeline unggah terbukti |
| **G2** | A → B, C | `folds_v2.csv` + `class_weights_v2.npy` | 16–17 Juli | B: retrain ConvNeXt v2 |
| **G3** | B → C, A | OOF v2 + checkpoint v2 (+ CV v2 vs v1) | 20 Juli | C: tuning prosedur final · B: lanjut multi-family |
| **G4** | B → C | OOF + checkpoint **per family** + config per model | 26–27 Juli | C: weight search → panen (≤28) → asuransi (≤29) |

Gate kebuka = owner kirim 1 pesan "GATE x hijau" di grup + artefak sudah di Drive dengan nama versi jelas.

---

## Peta Paralel (tidak ada yang idle)

```
            12–14 Juli            14–17 Juli            17–20 Juli           20–27 Juli            27–30 Juli
TRACK B  ██ baseline 5-fold ─G1─▶ eksperimen HP        ██ retrain v2 ─G3─▶  ██ multi-family ─G4─▶ bantu komposisi
            (prioritas #1)        (fold 0 saja)           (folds_v2)           (Swin→EffNetV2)      + log verifikasi

TRACK A  siapkan pipeline    ─G1─▶ ██ cleanlab +        ██ folds_v2 ─G2──▶   diagnosis OOF v2      bantu review
         cleanlab (dummy/          review visual           + weights_v2       (opsional, ringan)    + report
         OOF parsial fold-0)       (bertiga)

TRACK C  update validator,   ─G1─▶ ██ SAFETY NET ≤16 ·  tuning di OOF v1     ██ tuning OOF v2      ██ PANEN ≤28
         manifest, modul           bangun modul           (latihan prosedur)  ─G3─ prosedur kunci   ██ ASURANSI ≤29
         threshold (dummy)         threshold/weight

██ = pekerjaan utama fase itu · sisanya kerja paralel pengisi
```

**Intinya:** ketiga track selalu paralel. Yang berurutan hanya **artefaknya** (OOF → cleaning → retrain → multi-family → submission), bukan orangnya. Sambil menunggu gate, tiap track punya pekerjaan persiapan yang tidak bergantung siapa pun (baris "siapkan/bangun/update").

---

## Aturan Prioritas Kalau Jadwal Geser

1. Geser ≤ 2 hari → geser semua target berikutnya, tidak ada yang dipotong.
2. Geser > 2 hari → yang dikorbankan **jumlah family** (hybrid gugur duluan, lalu EffNetV2). Cleaning (A) dan disiplin submission (C) tidak pernah dikorbankan.
3. Deadline internal keras: **panen ≤ 28 Juli, asuransi ≤ 29 Juli**. 30 Juli hanya untuk keadaan darurat, bukan rencana.

---

## Ritme

- **Standup 10 menit tiap hari:** status gate (hijau/kuning/merah), blocker, siapa butuh GPU siapa.
- Pemegang GPU: B pakai Kaggle (kuota training utama); A & C di Colab/CPU.
- Semua artefak di Drive dengan nama berversi; kode di Git; notebook tipis.

---

*v2 — 12 Juli 2026. Menggantikan alur sinkron 3-hari di Workflow lama; prinsip anti-bentrok (contract-first, owner-only write, stub) tetap berlaku.*
