import os
import sys

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC_DIR))

# dataset.py adalah artefak Track A (read-only, lihat track_a/src/dataset.py).
# Di Colab file ini di-copy ke track_b/src; untuk test lokal kita import langsung
# dari track_a/src supaya tidak perlu duplikasi/copy manual.
TRACK_A_SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "track_a", "src")
sys.path.insert(0, os.path.abspath(TRACK_A_SRC_DIR))
