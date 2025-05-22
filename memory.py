import os
import sys
import time
import json
import csv
import sqlite3
import mmap
import tracemalloc
import array
import numpy as np
import tempfile

# -- Configuration -----------------------------------------------------------------
N = 1000000  # number of records for demo
OUTPUT_DIR = tempfile.mkdtemp(prefix="mem_demo_")

# -- Generate synthetic data --------------------------------------------------------
data = [(i, i * 0.5, f"text_{i % 1000}") for i in range(N)]

# -- Function to measure and print -----------------------------------------------
def measure(func, *args, **kwargs):
    """Utility to measure execution time and memory used."""
    tracemalloc.start()
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    duration = time.perf_counter() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, duration, peak

# -- 1) CSV ------------------------------------------------------------------------
csv_path = os.path.join(OUTPUT_DIR, "data.csv")
def write_csv(path, dataset):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "value", "text"])
        writer.writerows(dataset)

(_, csv_time, csv_mem) = measure(write_csv, csv_path, data)
csv_size = os.path.getsize(csv_path)

# -- 2) JSON -----------------------------------------------------------------------
json_path = os.path.join(OUTPUT_DIR, "data.json")
def write_json(path, dataset):
    with open(path, "w") as f:
        json.dump([{"id": i, "value": v, "text": t} for i, v, t in dataset], f)

(_, json_time, json_mem) = measure(write_json, json_path, data)
json_size = os.path.getsize(json_path)

# -- 3) SQLite ---------------------------------------------------------------------
sqlite_path = os.path.join(OUTPUT_DIR, "data.db")
def write_sqlite(path, dataset):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("CREATE TABLE demo(id INTEGER, value REAL, text TEXT)")
    c.executemany("INSERT INTO demo VALUES (?, ?, ?)", dataset)
    conn.commit()
    conn.close()

(_, sqlite_time, sqlite_mem) = measure(write_sqlite, sqlite_path, data)
sqlite_size = os.path.getsize(sqlite_path)

# -- 4) Python array.array (ints) --------------------------------------------------
def build_array(dataset):
    arr = array.array('I', (i for i, _, _ in dataset))
    return arr

(arr_obj, arr_time, arr_mem) = measure(build_array, data)
arr_size = arr_obj.buffer_info()[1] * arr_obj.itemsize  # number of elems * 4 bytes per uint32

# -- 5) NumPy array (floats) -------------------------------------------------------
def build_numpy(dataset):
    return np.fromiter((v for _, v, _ in dataset), dtype=np.float32)

(np_arr, np_time, np_mem) = measure(build_numpy, data)
np_size = np_arr.nbytes

# -- 6) Memory-map CSV -------------------------------------------------------------
def create_mmap(path):
    f = open(path, 'r+b')
    m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    return m

(mmap_obj, mmap_time, mmap_mem) = measure(create_mmap, csv_path)
mmap_size = sys.getsizeof(mmap_obj)  # overhead only

# -- Summarize results -------------------------------------------------------------
results = [
    ("CSV file", csv_size, csv_time, csv_mem),
    ("JSON file", json_size, json_time, json_mem),
    ("SQLite DB", sqlite_size, sqlite_time, sqlite_mem),
    ("array.array('I')", arr_size, arr_time, arr_mem),
    ("NumPy float32", np_size, np_time, np_mem),
    ("mmap object", mmap_size, mmap_time, mmap_mem),
]

# Pretty-print table
print(f"{'Type':25s} {'Disk/Memory Size':>15s} {'Time (s)':>10s} {'Peak Mem (KiB)':>15s}")
print("-" * 70)
for name, size, t, mem in results:
    print(f"{name:25s} {size/1024/1024:8.2f} MiB {t:10.3f} {mem/1024:15.2f} KiB")

# Provide explanatory summary:
print("\nExplanations:")
print("1. CSV and JSON: written to disk, JSON is more verbose hence larger & slower.")
print("2. SQLite: compact on-disk B-tree, write time includes index management.")
print("3. array.array vs NumPy: both dense; array uses pure Python, NumPy is C-backed.")
print("4. mmap: no data copy, small Python object overhead only; actual data remains on-disk mapped.")
