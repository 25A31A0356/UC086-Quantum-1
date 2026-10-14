"""
CLI Tool: Ingest and Cache TCGA-LUAD Cancer Genomics Dataset.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.tcga_loader import fetch_tcga_luad_cohort

def main():
    print("=" * 70)
    print("  TCGA-LUAD CANCER GENOMICS DATASET INGESTION (DOMAIN 2)")
    print("=" * 70)
    df = fetch_tcga_luad_cohort(cache_dir="data")
    print(f"[✓] Cohort size: {len(df)} primary tumor samples")
    print(f"[✓] Somatic EGFR mutants: {(df['EGFR_mutant'] == 1).sum()} ({(df['EGFR_mutant'] == 1).mean()*100:.1f}%)")
    print(f"[✓] Somatic KRAS mutants: {(df['KRAS_mutant'] == 1).sum()} ({(df['KRAS_mutant'] == 1).mean()*100:.1f}%)")
    print(f"[✓] Cached dataset saved to: data/tcga_luad_raw_cohort.csv")

if __name__ == "__main__":
    main()
