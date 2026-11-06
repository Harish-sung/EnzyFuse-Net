# EnzyFuse-Net V5: Evolutionary Scale Modeling (ESM-2) Transfer Learning for Multi-Class Enzyme Classification

**Authors:** Gopalsamy B, Sobana M*, Harish M, Kanagalakshmi S, Swathi S  
**Affiliation:** Dept. of Biotechnology, KIT - Kalaignar Karunanidhi Institute of Technology, Coimbatore, India  

## Repository Contents
1. `enzyfuse_v5_ieee_journal_colab.py` — Complete PyTorch 2.11.0+cu130 training, evaluation, and verification pipeline (8-Class `EC 1.*`–`EC 7.*` + `Non-Enzyme` and 6-Class `EC 1`–`EC 6`, `BLASTp` best-hit alignment, `CLEAN` supervised contrastive projection, and `<40%` homology-partitioned split).
2. `uniprot_enzyfuse_benchmark_accessions.csv` — Curated 4,000-sequence UniProtKB/Swiss-Prot benchmark manifest across all 8 functional classes (`EC 1`–`EC 7` Translocases + `Non-Enzyme` negative controls, 500 sequences/class, 80 <= L <= 400).
3. `uniprot_enzyfuse_6class_3000_accessions.csv` — Curated 3,000-sequence canonical enzyme subset (`EC 1`–`EC 6`, 500 sequences/class).

## Quick Start in Google Colab
```bash
# Print verified benchmark tables and parameter counts:
python enzyfuse_v5_ieee_journal_colab.py

# Run full live UniProt REST API curation, ESM-2 embedding extraction, and dual-split training:
python enzyfuse_v5_ieee_journal_colab.py --run_live
```
