# Workflow — Koordinasi Track A / B / C

**Proyek:** BDC Satria Data 2026 — Klasifikasi Citra Sampah
**Tujuan dokumen:** Menyatukan Track A (Data), B (Model), C (Evaluasi) supaya kerja paralel bertiga **tidak bentrok** — tidak saling nunggu, tidak saling menimpa file, tidak beda konvensi.

---

## 3 Prinsip Anti-Bentrok

1. **Contract-first.** Sepakati skema file & signature fungsi **sebelum ngoding** (8 Juli pagi). Semua kerja paralel menuju kontrak yang sama, jadi hasilnya nyambung tanpa nego ulang.
2. **Satu owner per artefak.** Hanya owner yang boleh menulis/overwrite sebuah artefak. Track lain **read-only**. Ini mencegah dua orang menimpa `folds.csv` atau checkpoint.
3. **Stub biar tak idle.** Track hilir (B, C) mulai kerja pakai **stub/dummy sesuai kontrak** sambil menunggu artefak asli. Tidak ada yang duduk diam nunggu track lain.

---

## Kontrak Antar-Track (Interface Freeze)

> Sepakati tabel ini di menit pertama, 8 Juli. Kalau salah satu kolom berubah kemudian → **umumkan ke semua** (lihat protokol breaking change).

| Artefak | Skema / Signature | Owner | Consumer |
|---------|-------------------|-------|----------|
| `folds.csv` | kolom: `filepath, label, fold` (fold 0–4) | A | B, C |
| `dataset.py` | `get_loaders(fold, img_size, batch)` → `(train_loader, val_loader)` + transform train/eval | A | B |
| `class_weights` | tensor `[3]`, urutan kelas 0/1/2 | A | B |
| test loader | loader test **terurut 1..1458** sesuai `submission.csv` | A | C |
| `fold{0..4}.pt` | checkpoint terbaik per fold (by val Macro-F1) | B | C |
| `oof.npy` | probabilitas `[N, 3]` + index cocok `folds.csv` | B | C |
| `config` | `img_size`, `norm` (ImageNet), `seed`, mapping label | A+B sepakat | semua |
| `submission_NamaTim.csv` | `id, predicted` (0/1/2), 1458 baris | C | panitia |

---

## Konvensi Bersama (sekali sepakat, semua ikut)

| Item | Nilai | Catatan |
|------|-------|---------|
| Mapping label | `0=Recyclable, 1=Electronic, 2=Organic` | **Aturan keras** — assert di tiap track |
| Seed | `42` | Satu angka, dipakai A/B/C |
| Image size | `224` | Baseline; kalau naik, ubah di `config`, umumkan |
| Normalisasi | mean/std **ImageNet** | Hindari leakage stat + konsisten train↔test |
| Penamaan fold | `fold0 … fold4` | Konsisten di file & kode |
| Kode | file `.py` di **Git repo** | Notebook cuma tipis buat orchestrate |

---

## Alur Sinkron 3 Hari (siapa, kapan, gate mana)

```
             TRACK A (Data)          TRACK B (Model)           TRACK C (Evaluasi)
             ───────────────         ───────────────           ──────────────────
8 Juli   ┌─ [SEMUA] Sepakati kontrak + konvensi (meeting 30–60 mnt) ─┐
pagi     └───────────────────────────────────────────────────────────┘
         Download + mulai EDA        Scaffold loop (pakai stub)      Bikin validator format
                                                                     + skeleton inference
8 Juli   EDA: distribusi + dupe ──▶ GATE 1 ──▶ Sanity overfit batch
sore     (hasil: weights, stats)                (loop terbukti benar)

9 Juli   folds.csv + dataloader ──▶ GATE 2 ──▶ Full 5-fold training
         final (lolos 1-batch)                 ──▶ OOF + CV ──▶ GATE 3 ──▶ Threshold tuning (OOF)

10 Juli  (support/report)           CV final + checkpoint          Ensemble+TTA → submission
pagi                                                                ──▶ GATE 4 ──▶ UNGGAH (safety net)
         ┌─ [SEMUA] Tulis report bareng (deadline 10 Juli) ─┐
         └──────────────────────────────────────────────────┘

11 Juli  ┌─ [SEMUA] Review 1 dengan reviewer ahli ─┐
         └─────────────────────────────────────────┘
```

---

## Handoff Gates (Definition of Ready)

Sebuah gate **kebuka** hanya kalau semua syaratnya terpenuhi. Track hilir tidak mulai sebelum gate-nya hijau.

| Gate | Dari → Ke | Syarat buka |
|------|-----------|-------------|
| **GATE 1** | A → B | Distribusi kelas diketahui · `class_weights` jadi · `dataset.py` stub sesuai signature |
| **GATE 2** | A → B | `folds.csv` final · dataloader lolos tes 1-batch · assert mapping 0/1/2 lolos |
| **GATE 3** | B → C | `oof.npy` menutupi seluruh data latih · 5 checkpoint tersimpan · `config` inference jelas |
| **GATE 4** | C → panitia | **Validator format lolos** · nama file benar · encoding 0/1/2 dicek |

---

## Peta Alur Data (satu gambar)

```
[A] folds.csv ─▶ [A] dataset.py ─▶ [B] training 5-fold ─▶ [B] checkpoints + OOF
                                                                  │
                                                    ┌─────────────┴─────────────┐
                                                    ▼                           ▼
                                        [C] threshold tuning (OOF)     [C] ensemble+TTA (test)
                                                    └─────────────┬─────────────┘
                                                                  ▼
                                            [C] apply thresholds ─▶ submission.csv ─▶ validator ─▶ UNGGAH
```

Selama semua pegang `folds.csv` yang sama + mapping 0/1/2 konsisten, integrasi mulus.

---

## Mekanisme Kolaborasi (anti-konflik teknis)

- **Kode di Git repo, bukan di notebook.** `dataset.py`, `train.py`, `eval.py` di GitHub. Notebook Kaggle/Colab cuma tipis buat panggil fungsi. (Notebook diedit barengan = konflik merge parah.)
- **Artefak besar di storage bersama.** `folds.csv`, checkpoint, `oof.npy` → simpan di **Kaggle Dataset privat** / Google Drive dengan versi jelas. Jangan kirim lewat chat.
- **Jadwal GPU Kaggle.** Kuota 30 jam/minggu → jangan rebutan. Training utama (B) di Kaggle; EDA (A) & eksperimen tambahan (C) di **Colab** biar hemat.
- **Owner-only write.** Cuma owner yang commit/overwrite artefaknya. Yang lain baca.

---

## Protokol Breaking Change

Kalau kontrak/konvensi harus berubah (mis. `img_size` 224→256, atau skema `folds.csv` nambah kolom):

1. **Umumkan dulu** ke grup sebelum ubah.
2. Update `config` / tabel kontrak di dokumen ini.
3. Track hilir konfirmasi siap sebelum owner commit perubahan.

Jangan ubah diam-diam — satu perubahan senyap bisa bikin OOF & submission tidak sinkron.

---

## Ritme Komunikasi

- **Standup harian** (8/9/10 Juli, ~10 menit): status, blocker, gate mana yang kebuka hari ini.
- **Titik integrasi = titik ngobrol.** Setiap gate = 1 pesan singkat "GATE x hijau, silakan lanjut".

---

## Risiko Koordinasi & Mitigasi

| Risiko | Dampak | Mitigasi |
|--------|--------|----------|
| Dua orang menimpa `folds.csv`/checkpoint | Tinggi | Owner-only write + Git |
| B/C idle nunggu A | Sedang | Contract-first + stub |
| Rebutan GPU Kaggle | Sedang | Jadwal + offload EDA/eksperimen ke Colab |
| Kontrak berubah senyap → OOF & test tak sinkron | Tinggi | Protokol breaking change |
| Mapping label beda antar track | Tinggi | Satu konstanta di `config` + assert di tiap track |
| Notebook diedit barengan → konflik | Sedang | Kode di `.py`/Git, notebook tipis |

---

## Checklist Kickoff (8 Juli, sebelum ngoding)

- [ ] Meeting kontrak + konvensi selesai, semua paham perannya
- [ ] Tabel Interface Freeze disepakati & dicatat
- [ ] Konvensi (mapping 0/1/2, seed 42, img 224, ImageNet norm) dikunci
- [ ] Git repo dibuat + struktur folder disepakati
- [ ] Storage bersama (Kaggle Dataset privat / Drive) disiapkan
- [ ] Jadwal GPU (siapa Kaggle, siapa Colab) ditetapkan
- [ ] Owner tiap artefak jelas
- [ ] Jam standup harian disepakati

---

*Dokumen ini adalah lem antar Track A/B/C. Kalau bingung "siapa yang pegang X" atau "boleh mulai belum" — jawabannya ada di tabel Kontrak & Gates di atas.*
