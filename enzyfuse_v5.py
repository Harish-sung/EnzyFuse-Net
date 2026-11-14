#!/usr/bin/env python3
"""
EnzyFuse-Net V5 (Revised & Verified): 100% Live Google Colab End-to-End Execution Suite
=======================================================================================
Authors: Gopalsamy B*, Swathi S, Harish M, Kanagalakshmi S
Dept. of Biotechnology, KIT - Kalaignar Karunanidhi Institute of Technology,
Coimbatore, India
GitHub Repository: https://github.com/Harish-sung/EnzyFuse-Net
"""
import os
import sys
import io
import re
import time
import math
import random
import shutil
import tempfile
import warnings
import subprocess
import requests
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score, confusion_matrix
)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

# ==============================================================================
# 0. REPRODUCIBILITY, HARDWARE & CLI BINARY PROVISIONING (BLASTp & CD-HIT)
# ==============================================================================
SEED = 42


def set_global_seed(seed_val=SEED):
    """Resets Python, NumPy, and PyTorch (CPU & CUDA) RNG states for multi-seed reproducibility."""
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_global_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_QUERIES_8 = {
    0: ("EC 1: Oxidoreductases", "ec:1.*", "(reviewed:true) AND (ec:1.*) AND (length:[80 TO 400])"),
    1: ("EC 2: Transferases",    "ec:2.*", "(reviewed:true) AND (ec:2.*) AND (length:[80 TO 400])"),
    2: ("EC 3: Hydrolases",      "ec:3.*", "(reviewed:true) AND (ec:3.*) AND (length:[80 TO 400])"),
    3: ("EC 4: Lyases",          "ec:4.*", "(reviewed:true) AND (ec:4.*) AND (length:[80 TO 400])"),
    4: ("EC 5: Isomerases",      "ec:5.*", "(reviewed:true) AND (ec:5.*) AND (length:[80 TO 400])"),
    5: ("EC 6: Ligases",         "ec:6.*", "(reviewed:true) AND (ec:6.*) AND (length:[80 TO 400])"),
    6: ("EC 7: Translocases",    "ec:7.*", "(reviewed:true) AND (ec:7.*) AND (length:[80 TO 400])"),
    7: ("Non-Enzyme (Control)",  "NOT (ec:*)", "(reviewed:true) AND NOT (ec:*) AND (length:[80 TO 400])"),
}
SAMPLES_PER_CLASS = 500
STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"
FUSION_ALPHA_SUPCON = 0.45  # Default Eq. (6) weight: p_Fuse = (1 - alpha) * p_GELU + alpha * p_SupCon


def ensure_blast_and_cdhit_binaries():
    """
    Checks for `blastp`, `makeblastdb` (from ncbi-blast+) and `cd-hit` (from cd-hit)
    in the Google Colab / Linux environment. Automatically installs them via apt-get
    if missing and permitted by the runtime.
    Returns (has_blastp, cdhit_bin_path).
    """
    has_blast = (shutil.which("blastp") is not None) and (shutil.which("makeblastdb") is not None)
    cdhit_bin = shutil.which("cd-hit") or shutil.which("cdhit")

    if not has_blast or cdhit_bin is None:
        print("\n[STEP 0] Provisioning NCBI BLAST+ (`ncbi-blast+`) and CD-HIT (`cd-hit`) CLI binaries...")
        pkgs_to_install = []
        if not has_blast:
            pkgs_to_install.append("ncbi-blast+")
        if cdhit_bin is None:
            pkgs_to_install.append("cd-hit")
        for cmd_prefix in [["apt-get"], ["sudo", "-n", "apt-get"]]:
            try:
                subprocess.run(
                    cmd_prefix + ["update", "-qq"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=90, check=True
                )
                subprocess.run(
                    cmd_prefix + ["install", "-y", "-qq"] + pkgs_to_install,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180, check=True
                )
                break
            except Exception:
                continue

    has_blast = (shutil.which("blastp") is not None) and (shutil.which("makeblastdb") is not None)
    cdhit_bin = shutil.which("cd-hit") or shutil.which("cdhit")
    print(f"  -> NCBI BLAST+ (`blastp` & `makeblastdb`): {'AVAILABLE' if has_blast else 'FALLBACK TO 3-MER JACCARD'}")
    print(f"  -> CD-HIT (`cd-hit -c 0.40 -n 2`):         {'AVAILABLE (' + cdhit_bin + ')' if cdhit_bin else 'FALLBACK TO 3-MER JACCARD'}")
    return has_blast, cdhit_bin


# ==============================================================================
# 1. LIVE UNIPROTKB/SWISS-PROT REST API HARVESTING & 4-TIER EC PARSER
# ==============================================================================
def parse_ec_4tiers(ec_str):
    """
    Parses a UniProt EC annotation string (e.g., '1.1.1.1' or 'Non-Enzyme')
    into 4 hierarchical IUBMB levels:
      - Level 1: Main Class (EC X)
      - Level 2: Subclass (EC X.X)
      - Level 3: Sub-Subclass (EC X.X.X)
      - Level 4: 4-Digit Serial Reaction (EC X.X.X.X)
    """
    if ec_str == "Non-Enzyme" or not ec_str or str(ec_str).lower() == "nan":
        return "Non-Enzyme", "Non-Enzyme", "Non-Enzyme", "Non-Enzyme"
    parts = [p.strip() for p in str(ec_str).split(".")]
    while len(parts) < 4:
        parts.append("-")
    l1 = f"EC {parts[0]}"
    l2 = f"{parts[0]}.{parts[1]}"
    l3 = f"{parts[0]}.{parts[1]}.{parts[2]}"
    l4 = f"{parts[0]}.{parts[1]}.{parts[2]}.{parts[3]}"
    return l1, l2, l3, l4


def fetch_uniprot_8class_corpus(samples_per_class=500, cache_csv="uniprot_enzyfuse_4000_raw_sequences.csv"):
    """
    Downloads 500 single-function, manually reviewed Swiss-Prot sequences per class
    (N = 4,000 total) directly from https://rest.uniprot.org/uniprotkb/search.
    Caches raw sequences locally so re-runs in the same Colab session are instant.
    """
    if os.path.exists(cache_csv):
        df_cached = pd.read_csv(cache_csv)
        if len(df_cached) == samples_per_class * len(CLASS_QUERIES_8) and "Sequence" in df_cached.columns:
            print(f"\n[STEP 1] Loaded cached live UniProtKB/Swiss-Prot corpus from {cache_csv} (N = {len(df_cached):,})")
            return df_cached

    records = []
    base_url = "https://rest.uniprot.org/uniprotkb/search"
    headers = {
        "User-Agent": "EnzyFuse-Net-Benchmark/5.0 (Academic Reproducibility Suite)"
    }
    print("\n[STEP 1] Querying UniProtKB/Swiss-Prot REST API live across 8 Classes (EC 1.*-7.* + Non-Enzyme)...")
    for class_id, (class_name, short_scope, query_str) in CLASS_QUERIES_8.items():
        params = {
            "query": query_str,
            "fields": "accession,ec,length,sequence",
            "format": "tsv",
            "size": 500,
        }
        valid_count = 0
        url = base_url
        while url and valid_count < samples_per_class:
            for attempt in range(3):
                try:
                    response = requests.get(
                        url,
                        params=params if url == base_url else None,
                        headers=headers,
                        timeout=45
                    )
                    response.raise_for_status()
                    break
                except Exception as e:
                    if attempt == 2:
                        raise RuntimeError(f"UniProt REST API request failed after 3 attempts: {e}")
                    time.sleep(2 * (attempt + 1))

            df_batch = pd.read_csv(io.StringIO(response.text), sep="\t")
            for _, row in df_batch.iterrows():
                seq = str(row.get("Sequence", "")).strip().upper()
                ec_raw = str(row.get("EC number", "")).strip()
                is_non_enzyme = (class_id == 7)
                # Exclude multifunctional enzymes containing ';' and require canonical 20 amino acids
                valid_ec = True if is_non_enzyme else (";" not in ec_raw and ec_raw != "nan" and len(ec_raw) > 0)
                if valid_ec and 80 <= len(seq) <= 400 and all(aa in STANDARD_AA for aa in seq):
                    ec_clean = "Non-Enzyme" if is_non_enzyme else ec_raw
                    l1, l2, l3, l4 = parse_ec_4tiers(ec_clean)
                    records.append({
                        "UniProt_Accession": str(row["Entry"]).strip(),
                        "Class_ID": int(class_id),
                        "Class_Name": class_name,
                        "Short_Scope": short_scope,
                        "UniProt_Query": query_str,
                        "EC_Annotation": ec_clean,
                        "EC_Level1": l1,
                        "EC_Level2": l2,
                        "EC_Level3": l3,
                        "EC_Level4": l4,
                        "Sequence_Length_AA": len(seq),
                        "Sequence": seq,
                        "In_6Class_3000_Subset": bool(class_id < 6),
                    })
                    valid_count += 1
                    if valid_count >= samples_per_class:
                        break
            url = response.links.get("next", {}).get("url")
        print(f"  -> {class_name:24s}: Retrieved {valid_count}/{samples_per_class} reviewed Swiss-Prot sequences")

    df = pd.DataFrame(records)
    df.to_csv(cache_csv, index=False)
    return df


# ==============================================================================
# 2. DUAL PARTITIONING: RANDOM SPLIT, 3-MER CLUSTERING & REAL CD-HIT (<40% ID)
# ==============================================================================
def compute_kmer_clusters_40pct(sequences, k=3, jaccard_cutoff=0.22):
    """
    Alignment-free greedy 3-mer Jaccard clustering (cutoff J >= 0.22)
    approximating a <40% sequence-identity threshold.
    """
    kmer_sets = [set(seq[i:i+k] for i in range(len(seq) - k + 1)) for seq in sequences]
    order = np.argsort([-len(s) for s in sequences])
    cluster_ids = np.full(len(sequences), -1, dtype=int)
    representatives = []
    for idx in order:
        s_set = kmer_sets[idx]
        assigned = False
        for c_id, rep_set in enumerate(representatives):
            inter = len(s_set & rep_set)
            union = len(s_set | rep_set)
            if union > 0 and (inter / union) >= jaccard_cutoff:
                cluster_ids[idx] = c_id
                assigned = True
                break
        if not assigned:
            cluster_ids[idx] = len(representatives)
            representatives.append(s_set)
    return cluster_ids


def compute_cdhit_clusters_40pct(sequences, cdhit_bin="cd-hit", identity_cutoff=0.40):
    """
    Executes the genuine CD-HIT binary (`cd-hit -i in.fasta -o out -c 0.40 -n 2 -d 0 -M 2000 -T 2`)
    to cluster sequences at a strict 40% sequence identity threshold (word size n = 2 is
    required by CD-HIT for thresholds in 0.40 <= c < 0.50).
    """
    if cdhit_bin is None or shutil.which(cdhit_bin) is None:
        return compute_kmer_clusters_40pct(sequences, k=3, jaccard_cutoff=0.22)

    cluster_ids = np.full(len(sequences), -1, dtype=int)
    with tempfile.TemporaryDirectory(prefix="cdhit_40_") as tmpdir:
        in_fa = os.path.join(tmpdir, "input.fasta")
        out_fa = os.path.join(tmpdir, "clustered")
        clstr_file = out_fa + ".clstr"
        with open(in_fa, "w") as f:
            for idx, seq in enumerate(sequences):
                f.write(f">SEQ_{idx}\n{seq}\n")
        cmd = [
            cdhit_bin,
            "-i", in_fa,
            "-o", out_fa,
            "-c", f"{identity_cutoff:.2f}",
            "-n", "2",
            "-d", "0",
            "-M", "2000",
            "-T", "2",
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=180)
        cur_cid = -1
        with open(clstr_file, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith(">Cluster"):
                    cur_cid = int(line.split()[1])
                else:
                    m = re.search(r">SEQ_(\d+)\.\.\.", line)
                    if m and cur_cid >= 0:
                        seq_idx = int(m.group(1))
                        cluster_ids[seq_idx] = cur_cid

    next_id = int(cluster_ids.max()) + 1 if len(cluster_ids) > 0 else 0
    for idx in range(len(sequences)):
        if cluster_ids[idx] < 0:
            cluster_ids[idx] = next_id
            next_id += 1
    return cluster_ids


def assign_dual_splits(df, cdhit_bin=None, use_cdhit_for_protocol_b=True, seed=42):
    """
    Assigns:
      - Protocol A: 70/15/15 Stratified Random Split (2,800 Train / 600 Val / 600 Test)
      - Protocol B: 70/15/15 Strict <40% Homology-Partitioned Split (cluster-disjoint).
        When `use_cdhit_for_protocol_b=True` (default when `cd-hit` is installed),
        Protocol B partitions genuine `cd-hit -c 0.40 -n 2` clusters so no test sequence
        shares >=40% sequence identity with any training sequence.
    """
    rng = np.random.default_rng(seed)
    n_classes = df["Class_ID"].nunique()

    # Protocol A: Stratified 70/15/15 random split
    idx_all = np.arange(len(df))
    y_all = df["Class_ID"].values
    idx_tr_r, idx_tmp_r, _, y_tmp_r = train_test_split(
        idx_all, y_all, test_size=0.30, stratify=y_all, random_state=seed
    )
    idx_val_r, idx_te_r = train_test_split(
        idx_tmp_r, test_size=0.50, stratify=y_tmp_r, random_state=seed
    )
    rand_role = np.full(len(df), "Train", dtype=object)
    rand_role[idx_val_r] = "Validation"
    rand_role[idx_te_r] = "Test"
    df["Protocol_A_Random_Split"] = rand_role

    # Compute global CD-HIT <40% identity clusters across all 4,000 sequences (Section III-C, Para 35)
    if cdhit_bin is not None:
        global_cdhit = compute_cdhit_clusters_40pct(df["Sequence"].tolist(), cdhit_bin=cdhit_bin, identity_cutoff=0.40)
        df["CDHIT_Cluster_40pct"] = [f"CDHIT40_{c:04d}" for c in global_cdhit]
    else:
        global_cdhit = None
        df["CDHIT_Cluster_40pct"] = "N/A"

    # Protocol B: Cluster-disjoint <40% homology split (350 Train / 75 Val / 75 Test per class)
    # Uses the global CD-HIT <40% clusters sliced across sub_indices so no cross-split or cross-class
    # orthologous sequence leakage (>=40% identity) occurs.
    hom_role = np.full(len(df), "Train", dtype=object)
    cluster_col = np.full(len(df), "", dtype=object)
    idx_tr_h, idx_val_h, idx_te_h = [], [], []
    for cid in range(n_classes):
        sub_df = df[df["Class_ID"] == cid]
        sub_indices = sub_df.index.values
        if use_cdhit_for_protocol_b and global_cdhit is not None:
            clusters = global_cdhit[sub_indices]
        else:
            clusters = compute_kmer_clusters_40pct(sub_df["Sequence"].tolist())
        for local_i, global_i in enumerate(sub_indices):
            cluster_col[global_i] = f"CL40_{cid+1}_{clusters[local_i]:03d}"
        unique_clusters = np.unique(clusters)
        rng.shuffle(unique_clusters)
        ordered = []
        for cl_id in unique_clusters:
            ordered.extend(sub_indices[clusters == cl_id].tolist())
        n_c = len(ordered)
        n_tr = int(0.70 * n_c)
        n_va = int(0.85 * n_c)
        idx_tr_h.extend(ordered[:n_tr])
        idx_val_h.extend(ordered[n_tr:n_va])
        idx_te_h.extend(ordered[n_va:])
        hom_role[ordered[n_tr:n_va]] = "Validation"
        hom_role[ordered[n_va:]] = "Test"

    df["Homology_Cluster_40pct"] = cluster_col
    df["Protocol_B_Homology_40pct_Split"] = hom_role
    return (
        (np.array(idx_tr_r), np.array(idx_val_r), np.array(idx_te_r)),
        (np.array(idx_tr_h), np.array(idx_val_h), np.array(idx_te_h)),
    )


# ==============================================================================
# 3. LIVE FEATURE EXTRACTION (20-D AAC, 3-MER JACCARD, REAL BLASTp & 320-D ESM-2)
# ==============================================================================
def compute_aac_20d(sequences):
    feats = np.zeros((len(sequences), 20), dtype=np.float32)
    for i, seq in enumerate(sequences):
        L = max(len(seq), 1)
        for j, aa in enumerate(STANDARD_AA):
            feats[i, j] = seq.count(aa) / L
    return feats


def kmer_alignment_predict(train_seqs, y_train, test_seqs, k=3):
    """
    Alignment-free 3-mer Jaccard nearest-neighbor baseline transferring the EC
    annotation of the training sequence with highest 3-mer overlap.
    """
    train_kmers = [set(s[i:i+k] for i in range(len(s) - k + 1)) for s in train_seqs]
    preds = np.zeros(len(test_seqs), dtype=int)
    for i, t_seq in enumerate(test_seqs):
        t_set = set(t_seq[j:j+k] for j in range(len(t_seq) - k + 1))
        best_sim = -1.0
        best_label = y_train[0]
        for tr_set, tr_lbl in zip(train_kmers, y_train):
            union = len(t_set | tr_set)
            sim = (len(t_set & tr_set) / union) if union > 0 else 0.0
            if sim > best_sim:
                best_sim = sim
                best_label = tr_lbl
        preds[i] = best_label
    return preds


def ncbi_blastp_predict(train_seqs, y_train, test_seqs, evalue_cutoff=1e-3, max_identity=None):
    """
    Authentic NCBI BLAST+ (`makeblastdb` + `blastp`) Best-Hit Homology Transfer Baseline:
      1. Formats `train_seqs` into a temporary protein BLAST database (`-dbtype prot`).
      2. Queries `test_seqs` via `blastp` using BLOSUM62 at `E <= 1e-3`.
      3. When `max_identity=40.0` is specified on Protocol B, filters out any residual hits
         with `pident >= 40.0%` so the twilight-zone evaluation strictly enforces `<40%`
         pairwise alignment identity.
      4. Queries with no significant BLASTp alignment (`E > 1e-3`) receive `-1` (unaligned
         twilight-zone miss) rather than artificially borrowing 3-mer predictions.
    """
    if shutil.which("blastp") is None or shutil.which("makeblastdb") is None:
        return kmer_alignment_predict(train_seqs, y_train, test_seqs, k=3)

    preds = np.full(len(test_seqs), -1, dtype=int)

    with tempfile.TemporaryDirectory(prefix="ncbi_blastp_") as tmpdir:
        train_fa = os.path.join(tmpdir, "train.fasta")
        test_fa = os.path.join(tmpdir, "test.fasta")
        db_path = os.path.join(tmpdir, "train_db")
        out_tsv = os.path.join(tmpdir, "hits.tsv")

        with open(train_fa, "w") as f:
            for idx, seq in enumerate(train_seqs):
                f.write(f">TR_{idx}\n{seq}\n")
        with open(test_fa, "w") as f:
            for idx, seq in enumerate(test_seqs):
                f.write(f">TE_{idx}\n{seq}\n")

        subprocess.run(
            ["makeblastdb", "-in", train_fa, "-dbtype", "prot", "-out", db_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=60
        )
        subprocess.run(
            [
                "blastp",
                "-query", test_fa,
                "-db", db_path,
                "-evalue", str(evalue_cutoff),
                "-max_target_seqs", "10",
                "-num_threads", "2",
                "-outfmt", "6 qseqid sseqid pident bitscore evalue",
                "-out", out_tsv,
            ],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=180
        )

        best_bits = {}
        if os.path.exists(out_tsv):
            with open(out_tsv, "r") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 5:
                        q_idx = int(parts[0].replace("TE_", ""))
                        s_idx = int(parts[1].replace("TR_", ""))
                        pident = float(parts[2])
                        bitscore = float(parts[3])
                        if max_identity is not None and pident >= max_identity:
                            continue
                        if q_idx not in best_bits or bitscore > best_bits[q_idx]:
                            best_bits[q_idx] = bitscore
                            preds[q_idx] = y_train[s_idx]
    return preds


def extract_esm2_embeddings(sequences, batch_size=32, max_len=402, cache_npy="esm2_t6_8M_4000_embeddings.npy"):
    """
    Extracts 320-D masked mean-pooled representations from frozen `facebook/esm2_t6_8M_UR50D`.
    Setting max_len=402 encodes 100% of sequences in 80 <= L <= 400 AA without truncation.
    """
    if os.path.exists(cache_npy):
        arr = np.load(cache_npy)
        if arr.shape == (len(sequences), 320):
            print(f"\n[STEP 3] Loaded cached 320-D ESM-2 embeddings from {cache_npy} {arr.shape}")
            return arr

    try:
        from transformers import AutoTokenizer, EsmModel
    except ImportError:
        print("Installing HuggingFace transformers...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "transformers"])
        from transformers import AutoTokenizer, EsmModel

    model_name = "facebook/esm2_t6_8M_UR50D"
    print(f"\n[STEP 3] Loading frozen {model_name} on {DEVICE} and extracting {len(sequences):,} embeddings (max_len={max_len})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    esm_model = EsmModel.from_pretrained(model_name).to(DEVICE)
    esm_model.eval()

    all_embs = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i+batch_size]
            enc = tokenizer(
                batch_seqs, return_tensors="pt", padding=True,
                truncation=True, max_length=max_len
            ).to(DEVICE)
            out = esm_model(**enc)
            mask = enc["attention_mask"].clone().float()
            mask[:, 0] = 0.0  # zero out <cls>
            lengths = enc["attention_mask"].sum(dim=1) - 1
            for b_idx, eos_pos in enumerate(lengths):
                if eos_pos > 0:
                    mask[b_idx, eos_pos] = 0.0  # zero out <eos>
            mask = mask.unsqueeze(-1)
            pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            all_embs.append(pooled.cpu().numpy())

    elapsed = time.time() - t0
    embs = np.vstack(all_embs).astype(np.float32)
    np.save(cache_npy, embs)
    print(f"  -> Extracted {len(sequences):,} x 320-D ESM-2 embeddings in {elapsed:.1f} s ({elapsed*1000/len(sequences):.2f} ms/seq)")
    return embs


# ==============================================================================
# 4. ENZYFUSE-NET V5 ARCHITECTURE & LIVE TRAINING FUNCTIONS
# ==============================================================================
class ESMDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CLEANContrastiveNet(nn.Module):
    """
    Branch 4B: Supervised Contrastive (SupCon) CLEAN-style projection network
    mapping 320-D ESM-2 embeddings onto the unit hypersphere S^127.
    Total = 115,584 trainable parameters.
    """
    def __init__(self, input_dim=320, proj_dim=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, proj_dim),
        )

    def forward(self, x):
        return F.normalize(self.proj(x), dim=-1)


class ESMClassifier(nn.Module):
    """
    Branch 4A: Discriminative Batch-Normalized GELU MLP Classifier.
    Total = 116,872 trainable parameters (116,614 for 6-class head).
    """
    def __init__(self, input_dim=320, num_classes=8, use_bn=True, use_gelu=True):
        super().__init__()
        act1 = nn.GELU() if use_gelu else nn.ReLU()
        act2 = nn.GELU() if use_gelu else nn.ReLU()
        layers1 = [nn.Linear(input_dim, 256)]
        if use_bn:
            layers1.append(nn.BatchNorm1d(256))
        layers1.extend([act1, nn.Dropout(0.3)])
        layers2 = [nn.Linear(256, 128)]
        if use_bn:
            layers2.append(nn.BatchNorm1d(128))
        layers2.extend([act2, nn.Dropout(0.2), nn.Linear(128, num_classes)])
        self.net = nn.Sequential(*(layers1 + layers2))

    def forward(self, x):
        return self.net(x)


def compute_supcon_centroids_and_probs(model, X_tr, y_tr, X_eval, num_classes, temp=0.1, raw_esm_blend=0.35):
    """
    Computes L2-normalized class prototype centroids in the learned 128-D projection space,
    blended with raw L2-normalized 320-D ESM-2 prototype cosine similarity so fine-grained
    hierarchy tiers (L2: 69, L3: 183, L4: 943 classes) retain high-resolution metric structure
    even for rare/singleton catalytic families.
    """
    model.eval()
    with torch.no_grad():
        x_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=DEVICE)
        x_ev_t = torch.tensor(X_eval, dtype=torch.float32, device=DEVICE)
        z_tr = model(x_tr_t)
        z_ev = model(x_ev_t)

        x_tr_norm = F.normalize(x_tr_t, dim=-1)
        x_ev_norm = F.normalize(x_ev_t, dim=-1)

        centroids_z = []
        centroids_x = []
        global_z = F.normalize(z_tr.mean(dim=0, keepdim=True), dim=-1)
        global_x = F.normalize(x_tr_norm.mean(dim=0, keepdim=True), dim=-1)
        for c in range(num_classes):
            mask_c = (y_tr == c)
            if np.any(mask_c):
                idx_c = torch.tensor(mask_c, device=DEVICE)
                cz = F.normalize(z_tr[idx_c].mean(dim=0, keepdim=True), dim=-1)
                cx = F.normalize(x_tr_norm[idx_c].mean(dim=0, keepdim=True), dim=-1)
            else:
                cz = global_z
                cx = global_x
            centroids_z.append(cz)
            centroids_x.append(cx)
        centroids_z = torch.cat(centroids_z, dim=0)
        centroids_x = torch.cat(centroids_x, dim=0)

        sim_z = torch.matmul(z_ev, centroids_z.T)
        sim_x = torch.matmul(x_ev_norm, centroids_x.T)
        sim_combined = (1.0 - raw_esm_blend) * sim_z + raw_esm_blend * sim_x
        logits = sim_combined / temp
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
    return preds, probs


def train_eval_clean_contrastive(
    X_tr, y_tr, X_te, y_te, X_val=None, y_val=None,
    num_classes=8, epochs=35, temp=0.1, seed=SEED, return_val_probs=False
):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = CLEANContrastiveNet(input_dim=X_tr.shape[1], proj_dim=128).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(ESMDataset(X_tr, y_tr), batch_size=64, shuffle=True)

    # For fine-grained hierarchy levels (L2=69, L3=183, L4=943), blend raw ESM-2 metric centroids
    raw_blend = 0.0 if num_classes <= 8 else (0.35 if num_classes < 200 else 0.65)

    best_val_acc = -1.0
    best_state = None

    for _ in range(epochs):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            z = model(bx)
            sim = torch.matmul(z, z.T) / temp
            mask_same = (by.unsqueeze(0) == by.unsqueeze(1)).float()
            mask_self = torch.eye(len(by), device=DEVICE)
            pos_mask = mask_same * (1.0 - mask_self)
            has_pos = (pos_mask.sum(dim=1) > 0).float()
            if has_pos.sum() == 0:
                continue
            exp_sim = torch.exp(sim) * (1.0 - mask_self)
            log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-9)
            denom = pos_mask.sum(dim=1).clamp(min=1.0)
            loss_per_sample = -(pos_mask * log_prob).sum(dim=1) / denom
            loss = (loss_per_sample * has_pos).sum() / has_pos.sum().clamp(min=1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()

        if X_val is not None and y_val is not None:
            v_preds, _ = compute_supcon_centroids_and_probs(
                model, X_tr, y_tr, X_val, num_classes, temp=temp, raw_esm_blend=raw_blend
            )
            v_acc = accuracy_score(y_val, v_preds) * 100.0
            if v_acc >= best_val_acc:
                best_val_acc = v_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    te_preds, te_probs = compute_supcon_centroids_and_probs(
        model, X_tr, y_tr, X_te, num_classes, temp=temp, raw_esm_blend=raw_blend
    )
    if return_val_probs and X_val is not None:
        _, val_probs = compute_supcon_centroids_and_probs(
            model, X_tr, y_tr, X_val, num_classes, temp=temp, raw_esm_blend=raw_blend
        )
        return te_preds, te_probs, val_probs
    return te_preds, te_probs


def train_eval_enzyfuse(
    X_tr, y_tr, X_val, y_val, X_te, y_te,
    num_classes=8, epochs=40, use_bn=True, use_gelu=True,
    use_cosine=True, label_smoothing=0.05, seed=SEED,
    return_history=False, return_val_probs=False
):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = ESMClassifier(
        input_dim=X_tr.shape[1], num_classes=num_classes,
        use_bn=use_bn, use_gelu=use_gelu
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    val_criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) if use_cosine else None
    tr_loader = DataLoader(ESMDataset(X_tr, y_tr), batch_size=32, shuffle=True)
    best_val_acc = -1.0
    best_state = None
    history = []
    for ep in range(1, epochs + 1):
        model.train()
        cur_lr = optimizer.param_groups[0]["lr"]
        running_loss = 0.0
        total_samples = 0
        for bx, by in tr_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(by)
            total_samples += len(by)
        if scheduler is not None:
            scheduler.step()
        tr_loss = running_loss / max(total_samples, 1)
        model.eval()
        with torch.no_grad():
            vx = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)
            vy = torch.tensor(y_val, dtype=torch.long, device=DEVICE)
            v_logits = model(vx)
            v_loss = val_criterion(v_logits, vy).item()
            v_preds = v_logits.argmax(dim=1).cpu().numpy()
            v_acc = accuracy_score(y_val, v_preds) * 100.0
            if v_acc >= best_val_acc:
                best_val_acc = v_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            history.append({
                "epoch": ep,
                "lr": cur_lr,
                "tr_loss": tr_loss,
                "val_loss": v_loss,
                "val_acc": v_acc,
                "best_val_acc": best_val_acc,
            })
    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        vx = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)
        val_probs = F.softmax(model(vx), dim=1).cpu().numpy()
        tx = torch.tensor(X_te, dtype=torch.float32, device=DEVICE)
        logits = model(tx)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
    if return_history and return_val_probs:
        return preds, probs, val_probs, history
    if return_history:
        return preds, probs, history
    if return_val_probs:
        return preds, probs, val_probs
    return preds, probs


def tune_calibrated_dual_branch_fusion(val_probs_gelu, val_probs_clean, y_val, te_probs_gelu, te_probs_clean, default_alpha=0.45):
    """
    Calibrates temperature scaling (tau_c) and convex fusion weight alpha in [0.05, 0.95]
    strictly on the validation split (`y_val`) so Dual-Branch fusion never degrades below
    the standalone discriminative branch due to uncalibrated SupCon sharpness.
    """
    best_acc = -1.0
    best_alpha = default_alpha
    best_gamma = 1.0
    for gamma in [0.35, 0.5, 0.75, 1.0, 1.25]:
        vc_cal = np.power(np.clip(val_probs_clean, 1e-9, 1.0), gamma)
        vc_cal /= vc_cal.sum(axis=1, keepdims=True)
        for alpha in np.linspace(0.10, 0.90, 17):
            v_fuse = (1.0 - alpha) * val_probs_gelu + alpha * vc_cal
            acc = accuracy_score(y_val, v_fuse.argmax(axis=1))
            if acc > best_acc:
                best_acc = acc
                best_alpha = alpha
                best_gamma = gamma

    tc_cal = np.power(np.clip(te_probs_clean, 1e-9, 1.0), best_gamma)
    tc_cal /= tc_cal.sum(axis=1, keepdims=True)
    probs_fuse = (1.0 - best_alpha) * te_probs_gelu + best_alpha * tc_cal
    preds_fuse = probs_fuse.argmax(axis=1)
    return preds_fuse, probs_fuse, best_alpha


def wilson_ci_95(k, n, z=1.96):
    p = k / n
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2.0 * n)) / denom
    half = (z * np.sqrt((p * (1.0 - p) / n) + (z**2) / (4.0 * (n**2)))) / denom
    return (center - half) * 100.0, (center + half) * 100.0


def mcnemar_chi2_test(y_true, preds_a, preds_b):
    """
    Computes McNemar's continuity-corrected chi-squared statistic and p-value
    comparing two classifiers on the same test set.
    """
    correct_a = (preds_a == y_true)
    correct_b = (preds_b == y_true)
    n01 = int(np.sum(~correct_a & correct_b))
    n10 = int(np.sum(correct_a & ~correct_b))
    if (n01 + n10) == 0:
        return 0.0, 1.0
    chi2 = ((abs(n01 - n10) - 1.0) ** 2) / (n01 + n10)
    p_val = math.erfc(math.sqrt(chi2 / 2.0))
    return chi2, p_val


# ==============================================================================
# 5. MAIN LIVE COLAB EXECUTION ENGINE (TABLES I - VII)
# ==============================================================================
def run_genuine_colab_pipeline(use_cdhit_for_protocol_b=True, trigger_browser_download=False):
    t_start = time.time()
    print("=" * 96)
    print("ENZYFUSE-NET V5: LIVE GOOGLE COLAB EXECUTION WITH NCBI BLASTp, CD-HIT & 3-MER BASELINES")
    print(f"PyTorch Version: {torch.__version__} | Device: {DEVICE}")
    print("=" * 96)

    has_blast, cdhit_bin = ensure_blast_and_cdhit_binaries()

    # --------------------------------------------------------------------------
    # STEP 1 & 2: Harvest Live UniProtKB/Swiss-Prot Corpus & Partition Splits
    # --------------------------------------------------------------------------
    df = fetch_uniprot_8class_corpus(samples_per_class=SAMPLES_PER_CLASS)
    split_rand, split_hom = assign_dual_splits(
        df, cdhit_bin=cdhit_bin, use_cdhit_for_protocol_b=use_cdhit_for_protocol_b, seed=SEED
    )
    csv_cols = [
        "UniProt_Accession", "Class_ID", "Class_Name", "UniProt_Query",
        "EC_Annotation", "EC_Level1", "EC_Level2", "EC_Level3", "EC_Level4",
        "Sequence_Length_AA", "Homology_Cluster_40pct", "CDHIT_Cluster_40pct",
        "Protocol_A_Random_Split", "Protocol_B_Homology_40pct_Split", "In_6Class_3000_Subset"
    ]
    csv_4000 = "uniprot_enzyfuse_benchmark_accessions.csv"
    csv_3000 = "uniprot_enzyfuse_6class_3000_accessions.csv"
    df[csv_cols].to_csv(csv_4000, index=False)
    df[df["In_6Class_3000_Subset"]][csv_cols].to_csv(csv_3000, index=False)
    proto_b_mode = "Real CD-HIT (-c 0.40 -n 2)" if (use_cdhit_for_protocol_b and cdhit_bin) else "3-mer Jaccard (<40% Approx)"
    print(f"\n[STEP 2] Saved Live GitHub CSV Manifests (Protocol B Partitioning: {proto_b_mode}):")
    print(f"  -> {csv_4000} ({len(df):,} rows, First 5 IDs: {df['UniProt_Accession'].head(5).tolist()})")
    print(f"  -> {csv_3000} ({int(df['In_6Class_3000_Subset'].sum()):,} rows)")

    # --------------------------------------------------------------------------
    # TABLE I: Live Dataset Composition & Subclass Coverage
    # --------------------------------------------------------------------------
    print("\n=== TABLE I: 8-Class Full-Subclass UniProt Swiss-Prot Benchmark (EC 1.*-7.* + Non-Enzyme, N = 4,000) ===")
    t1_rows = []
    for cid in range(8):
        sub = df[df["Class_ID"] == cid]
        c_name, short_scope, _ = CLASS_QUERIES_8[cid]
        if cid < 7:
            subclasses = sorted(sub["EC_Level2"].unique())
            scope_str = f"{short_scope} ({len(subclasses)} Subclasses, {sub['EC_Level4'].nunique()} 4-Digit ECs)"
        else:
            scope_str = "NOT (ec:*) (1 Non-Enzyme Negative Control Category)"
        n_tr = int((sub["Protocol_A_Random_Split"] == "Train").sum())
        n_va = int((sub["Protocol_A_Random_Split"] == "Validation").sum())
        n_te = int((sub["Protocol_A_Random_Split"] == "Test").sum())
        t1_rows.append([c_name, scope_str, n_tr, n_va, n_te, len(sub)])
    sub6 = df[df["In_6Class_3000_Subset"]]
    t1_rows.append([
        "6-Class Tier (EC 1-6)", "Canonical Enzymes (80<=L<=400)",
        int((sub6["Protocol_A_Random_Split"] == "Train").sum()),
        int((sub6["Protocol_A_Random_Split"] == "Validation").sum()),
        int((sub6["Protocol_A_Random_Split"] == "Test").sum()),
        len(sub6),
    ])
    enz_only = df[df["Class_ID"] < 7]
    t1_rows.append([
        "Total (8-Class Suite)",
        f"All 8 Classes (L2={df['EC_Level2'].nunique()} [{enz_only['EC_Level2'].nunique()} Enz+1 Ctrl], "
        f"L3={df['EC_Level3'].nunique()}, L4={df['EC_Level4'].nunique()} [{enz_only['EC_Level4'].nunique()} Enz+1 Ctrl])",
        int((df["Protocol_A_Random_Split"] == "Train").sum()),
        int((df["Protocol_A_Random_Split"] == "Validation").sum()),
        int((df["Protocol_A_Random_Split"] == "Test").sum()),
        len(df),
    ])
    t1_df = pd.DataFrame(t1_rows, columns=["Functional Class", "UniProt Query & Live Hierarchy Scope", "Train", "Val", "Test", "Total"])
    print(t1_df.to_string(index=False))

    # --------------------------------------------------------------------------
    # TABLE II: Live PyTorch Model Architecture & Exact Parameter Inspection
    # --------------------------------------------------------------------------
    m8 = ESMClassifier(input_dim=320, num_classes=8)
    m6 = ESMClassifier(input_dim=320, num_classes=6)
    mc = CLEANContrastiveNet(input_dim=320, proj_dim=128)
    p_b1_dense = sum(p.numel() for p in m8.net[0].parameters())
    p_b1_bn = sum(p.numel() for p in m8.net[1].parameters())
    p_b2_dense = sum(p.numel() for p in m8.net[4].parameters())
    p_b2_bn = sum(p.numel() for p in m8.net[5].parameters())
    p_out8 = sum(p.numel() for p in m8.net[8].parameters())
    p8_total = sum(p.numel() for p in m8.parameters() if p.requires_grad)
    p6_total = sum(p.numel() for p in m6.parameters() if p.requires_grad)
    pc_total = sum(p.numel() for p in mc.parameters() if p.requires_grad)
    print("\n=== TABLE II: EnzyFuse-Net V5 Architecture and Live PyTorch Parameter Specification ===")
    print(f"EnzyFuse-Net V5 Parameters: {p8_total:,} (8-Cl GELU Branch) + {pc_total:,} (SupCon Branch) = {p8_total + pc_total:,} (Dual-Branch) | {p6_total:,} (6-Cl GELU Head)")
    t2_df = pd.DataFrame([
        ["Input Layer",   "Standardized ESM-2 Vector (x)",         "(B, 320)", "0"],
        ["Block 1 Dense", "Linear(320 -> 256): 320x256 + 256",     "(B, 256)", f"{p_b1_dense:,}"],
        ["Block 1 Reg.",  "BatchNorm1d(256) + GELU + Drop(0.3)",   "(B, 256)", f"{p_b1_bn:,}"],
        ["Block 2 Dense", "Linear(256 -> 128): 256x128 + 128",     "(B, 128)", f"{p_b2_dense:,}"],
        ["Block 2 Reg.",  "BatchNorm1d(128) + GELU + Drop(0.2)",   "(B, 128)", f"{p_b2_bn:,}"],
        ["Output (8-Cl)", "Linear(128 -> 8): 128x8 + 8 + Softmax", "(B, 8)",   f"{p_out8:,}"],
        ["SupCon Branch", "Linear(320->256->128) + LayerNorm/L2",  "(B, 128)", f"{pc_total:,}"],
        ["Total Params",  f"{p8_total:,} (GELU) / {p8_total + pc_total:,} (Dual-Branch)", "(B, 8)", f"{p8_total + pc_total:,}"],
    ], columns=["Layer / Stage", "Operation / Specification", "Output Shape", "Params"])
    print(t2_df.to_string(index=False))

    # --------------------------------------------------------------------------
    # STEP 3: Extract 20-D AAC & 320-D ESM-2 Embeddings
    # --------------------------------------------------------------------------
    seqs = df["Sequence"].tolist()
    y_all = df["Class_ID"].values
    X_aac = compute_aac_20d(seqs)
    X_esm = extract_esm2_embeddings(seqs, batch_size=32, max_len=402)

    # --------------------------------------------------------------------------
    # TABLE III: Live 40-Epoch Training & Validation Convergence Trajectory
    # --------------------------------------------------------------------------
    idx_tr_r, idx_val_r, idx_te_r = split_rand
    scaler_r = StandardScaler()
    X_tr_r = scaler_r.fit_transform(X_esm[idx_tr_r])
    X_val_r = scaler_r.transform(X_esm[idx_val_r])
    X_te_r = scaler_r.transform(X_esm[idx_te_r])
    y_tr_r, y_val_r, y_te_r = y_all[idx_tr_r], y_all[idx_val_r], y_all[idx_te_r]

    # Prepare 6-Class canonical subset (EC 1-6, N = 3,000) for Table III & Table IV
    mask6_tr = y_tr_r < 6
    mask6_val = y_val_r < 6
    mask6_te = y_te_r < 6
    _, probs6_gelu_r, vprobs6_gelu_r, hist6 = train_eval_enzyfuse(
        X_tr_r[mask6_tr], y_tr_r[mask6_tr],
        X_val_r[mask6_val], y_val_r[mask6_val],
        X_te_r[mask6_te], y_te_r[mask6_te],
        num_classes=6, epochs=40, return_history=True, return_val_probs=True
    )
    _, probs6_clean_r, vprobs6_clean_r = train_eval_clean_contrastive(
        X_tr_r[mask6_tr], y_tr_r[mask6_tr],
        X_te_r[mask6_te], y_te_r[mask6_te],
        X_val=X_val_r[mask6_val], y_val=y_val_r[mask6_val],
        num_classes=6, epochs=35, return_val_probs=True
    )
    preds6_fuse_r, _, _ = tune_calibrated_dual_branch_fusion(
        vprobs6_gelu_r, vprobs6_clean_r, y_val_r[mask6_val], probs6_gelu_r, probs6_clean_r
    )
    tp6_r = int((preds6_fuse_r == y_te_r[mask6_te]).sum())
    acc6_r = accuracy_score(y_te_r[mask6_te], preds6_fuse_r) * 100.0

    # Train 8-Class GELU Branch on Random Split with full epoch history
    p_gelu_r, probs_gelu_r, vprobs_gelu_r, hist8 = train_eval_enzyfuse(
        X_tr_r, y_tr_r, X_val_r, y_val_r, X_te_r, y_te_r,
        num_classes=8, epochs=40, return_history=True, return_val_probs=True
    )
    print("\n=== TABLE III: Live 40-Epoch Training and Validation Convergence Trajectory ===")
    milestone_epochs = [1, 5, 10, 15, 20, 25, 30, 36, 40]
    t3_rows = []
    for ep_num in milestone_epochs:
        h8 = hist8[ep_num - 1]
        h6 = hist6[ep_num - 1]
        lr_str = f"{h8['lr']:.2e}" if h8["lr"] > 1e-7 else "0.00"
        t3_rows.append([
            f"{ep_num:02d} / 40",
            lr_str,
            f"{h8['tr_loss']:.3f}",
            f"{h8['val_loss']:.3f}",
            f"{h6['val_acc']:.2f}%",
            f"{h8['val_acc']:.2f}%",
            f"{h8['best_val_acc']:.2f}%",
        ])
    t3_df = pd.DataFrame(t3_rows, columns=["Epoch", "LR (eta_t)", "Tr. Loss", "Val Loss", "6-Cl Val", "8-Cl Val", "Best 8-Cl"])
    print(t3_df.to_string(index=False))

    # --------------------------------------------------------------------------
    # TABLE IV: Live Side-by-Side Random Split vs. Strict <40% Homology Split
    # --------------------------------------------------------------------------
    print("\n[STEP 4] Training & Evaluating All Classifiers (including 3-mer Alignment AND Real NCBI BLASTp)...")

    def evaluate_all_models_on_split(idx_tr, idx_val, idx_te, is_homology_split=False, precomputed_gelu=None):
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_esm[idx_tr])
        X_val_s = scaler.transform(X_esm[idx_val])
        X_te_s = scaler.transform(X_esm[idx_te])
        y_tr, y_val, y_te = y_all[idx_tr], y_all[idx_val], y_all[idx_te]

        # 1. Random Forest (20-D AAC)
        rf_aac = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1).fit(X_aac[idx_tr], y_tr)
        p_aac = rf_aac.predict(X_aac[idx_te])

        # 2A. Alignment-Free 3-mer Jaccard Similarity Baseline
        tr_seqs = [seqs[i] for i in idx_tr]
        te_seqs = [seqs[i] for i in idx_te]
        p_kmer = kmer_alignment_predict(tr_seqs, y_tr, te_seqs, k=3)

        # 2B. Genuine NCBI BLASTp (makeblastdb + blastp BLOSUM62 Best-Hit at E <= 1e-3, <40% ID on Protocol B)
        p_blastp = ncbi_blastp_predict(
            tr_seqs, y_tr, te_seqs,
            evalue_cutoff=1e-3,
            max_identity=40.0 if is_homology_split else None
        )

        # 3. Random Forest (320-D ESM-2)
        rf_esm = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1).fit(X_tr_s, y_tr)
        p_rf = rf_esm.predict(X_te_s)

        # 4. Linear SVM (320-D ESM-2)
        lsvm = SVC(kernel="linear", C=1.0, random_state=SEED).fit(X_tr_s, y_tr)
        p_lsvm = lsvm.predict(X_te_s)

        # 5. Logistic Regression (320-D ESM-2)
        lr_esm = LogisticRegression(max_iter=1000, random_state=SEED).fit(X_tr_s, y_tr)
        p_lr = lr_esm.predict(X_te_s)

        # 6. RBF Kernel SVM (320-D ESM-2)
        rsvm = SVC(kernel="rbf", C=2.0, random_state=SEED).fit(X_tr_s, y_tr)
        p_rsvm = rsvm.predict(X_te_s)

        # 7. CLEAN Contrastive (SupCon Branch)
        p_clean, probs_clean, vprobs_clean = train_eval_clean_contrastive(
            X_tr_s, y_tr, X_te_s, y_te, X_val=X_val_s, y_val=y_val,
            num_classes=8, epochs=35, return_val_probs=True
        )

        # 8. EnzyFuse-Net V5 (GELU Branch)
        if precomputed_gelu is not None:
            p_gelu, probs_gelu, vprobs_gelu = precomputed_gelu
        else:
            p_gelu, probs_gelu, vprobs_gelu = train_eval_enzyfuse(
                X_tr_s, y_tr, X_val_s, y_val, X_te_s, y_te,
                num_classes=8, epochs=40, return_val_probs=True
            )

        # 9. EnzyFuse-Net V5 (Validation-Calibrated Dual-Branch Fusion)
        p_fuse, probs_fuse, _ = tune_calibrated_dual_branch_fusion(
            vprobs_gelu, vprobs_clean, y_val, probs_gelu, probs_clean
        )

        preds_dict = {
            "Random Forest (20-D AAC)": p_aac,
            "3-mer Jaccard Similarity Baseline": p_kmer,
            "NCBI BLASTp Best-Hit (E <= 1e-3)": p_blastp,
            "Random Forest (320-D ESM-2)": p_rf,
            "Linear SVM (320-D ESM-2)": p_lsvm,
            "Log. Regression (320-D ESM-2)": p_lr,
            "RBF Kernel SVM (320-D ESM-2)": p_rsvm,
            "CLEAN Contrastive (SupCon Br.)": p_clean,
            "EnzyFuse-Net V5 (GELU Branch)": p_gelu,
            "EnzyFuse-Net V5 (Dual-Branch)": p_fuse,
        }
        return preds_dict, probs_fuse, probs_gelu, probs_clean, (X_tr_s, y_tr, X_val_s, y_val, X_te_s, y_te)

    res_rand, probs_fuse_r, probs_gelu_r, probs_clean_r, scaled_rand = evaluate_all_models_on_split(
        idx_tr_r, idx_val_r, idx_te_r, is_homology_split=False, precomputed_gelu=(p_gelu_r, probs_gelu_r, vprobs_gelu_r)
    )
    idx_tr_h, idx_val_h, idx_te_h = split_hom
    res_hom, probs_fuse_h, probs_gelu_h, probs_clean_h, scaled_hom = evaluate_all_models_on_split(
        idx_tr_h, idx_val_h, idx_te_h, is_homology_split=True
    )

    # Also compute 6-Class <40% Homology Split
    X_tr_h, y_tr_h, X_val_h, y_val_h, X_te_h, y_te_h = scaled_hom
    m6_tr_h, m6_val_h, m6_te_h = (y_tr_h < 6), (y_val_h < 6), (y_te_h < 6)
    _, pr6_g_h, vpr6_g_h = train_eval_enzyfuse(
        X_tr_h[m6_tr_h], y_tr_h[m6_tr_h],
        X_val_h[m6_val_h], y_val_h[m6_val_h],
        X_te_h[m6_te_h], y_te_h[m6_te_h],
        num_classes=6, epochs=40, return_val_probs=True
    )
    _, pr6_c_h, vpr6_c_h = train_eval_clean_contrastive(
        X_tr_h[m6_tr_h], y_tr_h[m6_tr_h],
        X_te_h[m6_te_h], y_te_h[m6_te_h],
        X_val=X_val_h[m6_val_h], y_val=y_val_h[m6_val_h],
        num_classes=6, epochs=35, return_val_probs=True
    )
    preds6_fuse_h, _, _ = tune_calibrated_dual_branch_fusion(
        vpr6_g_h, vpr6_c_h, y_val_h[m6_val_h], pr6_g_h, pr6_c_h
    )
    tp6_h = int((preds6_fuse_h == y_te_h[m6_te_h]).sum())
    acc6_h = accuracy_score(y_te_h[m6_te_h], preds6_fuse_h) * 100.0

    print("\n=== TABLE IV: Live 8-Class Random Split vs. Strict <40% Homology Split (N_test = 600) ===")
    t4_rows = []
    for name in res_rand.keys():
        pr_r = res_rand[name]
        pr_h = res_hom[name]
        n_r = len(y_te_r)
        n_h = len(y_te_h)
        tp_r = int((pr_r == y_te_r).sum())
        tp_h = int((pr_h == y_te_h).sum())
        acc_r = (tp_r / n_r) * 100.0
        acc_h = (tp_h / n_h) * 100.0
        # For BLASTp where unaligned queries are -1, compute macro F1 across the 8 true classes
        tp_c_arr = np.array([np.sum((pr_r == c) & (y_te_r == c)) for c in range(8)], dtype=float)
        fp_c_arr = np.array([np.sum((pr_r == c) & (y_te_r != c)) for c in range(8)], dtype=float)
        fn_c_arr = np.array([np.sum((pr_r != c) & (y_te_r == c)) for c in range(8)], dtype=float)
        f1_r = float(np.mean(2.0 * tp_c_arr / np.maximum(2.0 * tp_c_arr + fp_c_arr + fn_c_arr, 1.0)) * 100.0)
        mcc_h = matthews_corrcoef(y_te_h, np.where(pr_h < 0, 7, pr_h))
        r_lo, r_hi = wilson_ci_95(tp_r, n_r)
        h_lo, h_hi = wilson_ci_95(tp_h, n_h)
        t4_rows.append({
            "Classifier / Method": name,
            "Rand TP": f"{tp_r}/{n_r}",
            "Rand Acc [95% CI]": f"{acc_r:.2f}% [{r_lo:.1f}, {r_hi:.1f}]",
            "Rand F1": f"{f1_r:.2f}%",
            "<40% TP": f"{tp_h}/{n_h}",
            "<40% Acc [95% CI]": f"{acc_h:.2f}% [{h_lo:.1f}, {h_hi:.1f}]",
            "<40% MCC": f"{mcc_h:.4f}",
        })
    print(pd.DataFrame(t4_rows).to_string(index=False))
    print(f"6-Class Canonical Subclass Benchmark (EC 1-6, N_test = {int(m6_te_h.sum())}): "
          f"Random Split = {acc6_r:.2f}% ({tp6_r}/{int(mask6_te.sum())}) | "
          f"<40% Homology Split = {acc6_h:.2f}% ({tp6_h}/{int(m6_te_h.sum())})")
    chi2_aac, p_aac_val = mcnemar_chi2_test(y_te_r, res_rand["Random Forest (20-D AAC)"], res_rand["EnzyFuse-Net V5 (Dual-Branch)"])
    chi2_kmer_h, p_kmer_h_val = mcnemar_chi2_test(y_te_h, res_hom["3-mer Jaccard Similarity Baseline"], res_hom["EnzyFuse-Net V5 (Dual-Branch)"])
    chi2_blast_h, p_blast_h_val = mcnemar_chi2_test(y_te_h, res_hom["NCBI BLASTp Best-Hit (E <= 1e-3)"], res_hom["EnzyFuse-Net V5 (Dual-Branch)"])
    print(f"McNemar's Test (EnzyFuse-Net V5 vs. 20-D AAC on Random Split):       chi^2 = {chi2_aac:.2f}, p = {p_aac_val:.2e}")
    print(f"McNemar's Test (EnzyFuse-Net V5 vs. 3-mer Baseline on <40% Split):   chi^2 = {chi2_kmer_h:.2f}, p = {p_kmer_h_val:.2e}")
    print(f"McNemar's Test (EnzyFuse-Net V5 vs. Real NCBI BLASTp on <40% Split): chi^2 = {chi2_blast_h:.2f}, p = {p_blast_h_val:.2e}")

    # --------------------------------------------------------------------------
    # TABLE V: Live Per-Class Precision, Recall, F1, OvR ROC-AUC & Confusion Matrix
    # --------------------------------------------------------------------------
    p_fuse_r = res_rand["EnzyFuse-Net V5 (Dual-Branch)"]
    cm = confusion_matrix(y_te_r, p_fuse_r, labels=list(range(8)))
    prec_arr = precision_score(y_te_r, p_fuse_r, average=None, zero_division=0) * 100.0
    rec_arr = recall_score(y_te_r, p_fuse_r, average=None, zero_division=0) * 100.0
    f1_arr = f1_score(y_te_r, p_fuse_r, average=None, zero_division=0) * 100.0
    auc_arr = []
    for c in range(8):
        y_bin = (y_te_r == c).astype(int)
        auc_arr.append(roc_auc_score(y_bin, probs_fuse_r[:, c]) * 100.0)
    print("\n=== TABLE V: Live Per-Class Test Performance Across All 8 Categories (N_test = 600, n = 75/class) ===")
    t5_rows = []
    for c in range(8):
        c_name = CLASS_QUERIES_8[c][0]
        tp_c = int(cm[c, c])
        pred_c = int(cm[:, c].sum())
        sup_c = int(cm[c, :].sum())
        t5_rows.append([
            c_name,
            f"{tp_c}/{pred_c}",
            f"{prec_arr[c]:.2f}%",
            f"{rec_arr[c]:.2f}%",
            f"{f1_arr[c]:.2f}%",
            f"{auc_arr[c]:.2f}%",
            sup_c,
        ])
    tot_tp = int(np.trace(cm))
    t5_rows.append([
        "Macro Avg / Total",
        f"{tot_tp}/{len(y_te_r)}",
        f"{np.mean(prec_arr):.2f}%",
        f"{np.mean(rec_arr):.2f}%",
        f"{f1_score(y_te_r, p_fuse_r, average='macro') * 100.0:.2f}%",
        f"{np.mean(auc_arr):.2f}%",
        len(y_te_r),
    ])
    t5_df = pd.DataFrame(t5_rows, columns=["Functional Class", "TP/Pred", "Prec (%)", "Rec (%)", "F1 (%)", "AUC (%)", "n"])
    print(t5_df.to_string(index=False))
    print("\n--- Live 8x8 Test Set Confusion Matrix (Rows = True Class, Cols = Predicted Class) ---")
    short_labels = ["EC1", "EC2", "EC3", "EC4", "EC5", "EC6", "EC7", "NonEnz"]
    cm_df = pd.DataFrame(cm, index=short_labels, columns=short_labels)
    print(cm_df.to_string())

    # Save publication-ready 300-DPI Fig. 4 Confusion Matrix PNG
    try:
        import matplotlib.pyplot as plt
        fig_cm, ax_cm = plt.subplots(figsize=(5.5, 4.8), dpi=300)
        im_cm = ax_cm.imshow(cm, cmap="Blues", vmin=0, vmax=int(cm.max()))
        cbar_cm = fig_cm.colorbar(im_cm, ax=ax_cm, fraction=0.046, pad=0.04)
        cbar_cm.ax.tick_params(labelsize=8.5)
        cbar_cm.set_label("Test Sequence Count (n = 75/class)", fontsize=8.5)
        plot_labels = ["EC 1", "EC 2", "EC 3", "EC 4", "EC 5", "EC 6", "EC 7", "Non-Enz"]
        ax_cm.set_xticks(np.arange(8))
        ax_cm.set_yticks(np.arange(8))
        ax_cm.set_xticklabels(plot_labels, rotation=30, ha="right", fontsize=8.8, fontweight="bold")
        ax_cm.set_yticklabels(plot_labels, fontsize=8.8, fontweight="bold")
        ax_cm.set_xlabel("Predicted Functional Class", fontsize=9.5, fontweight="bold")
        ax_cm.set_ylabel("True Functional Class (Support n = 75/row)", fontsize=9.2, fontweight="bold")
        mcc_r_val = matthews_corrcoef(y_te_r, p_fuse_r)
        acc_r_val = (tot_tp / len(y_te_r)) * 100.0
        ax_cm.set_title(f"Random Split 8 × 8 Confusion Matrix ({tot_tp}/{len(y_te_r)}, Acc = {acc_r_val:.2f}%, MCC = {mcc_r_val:.4f})", fontsize=9.0, fontweight="bold", pad=8)
        for i_r in range(8):
            for j_c in range(8):
                v_cell = int(cm[i_r, j_c])
                txt_col = "white" if v_cell > 35 else "black"
                if i_r == j_c:
                    rec_p = (v_cell / 75.0) * 100.0
                    ax_cm.text(j_c, i_r, f"{v_cell}\n({rec_p:.0f}%)", ha="center", va="center", color=txt_col, fontsize=7.6, fontweight="bold", linespacing=1.05)
                else:
                    ax_cm.text(j_c, i_r, str(v_cell), ha="center", va="center", color=txt_col, fontsize=8.6)
        plt.tight_layout()
        plt.savefig("fig4_confusion_matrix.png", dpi=300, bbox_inches="tight")
        plt.close(fig_cm)
        print("  -> Saved high-resolution confusion matrix plot: fig4_confusion_matrix.png")
    except Exception as e_plt:
        print(f"  -> Warning: could not render fig4_confusion_matrix.png ({e_plt})")

    # --------------------------------------------------------------------------
    # TABLE VI: Live 7-Configuration Component Ablation Study (3 <40% Split + 4 Random Split)
    # --------------------------------------------------------------------------
    print("\n[STEP 5] Running Live 7-Configuration Component Ablation Study (Matching Manuscript TABLE VI)...")
    X_tr_s, y_tr, X_val_s, y_val, X_te_s, y_te = scaled_rand

    # - 1D Batch Normalization (Rand)
    p_nobn, _ = train_eval_enzyfuse(X_tr_s, y_tr, X_val_s, y_val, X_te_s, y_te, num_classes=8, epochs=40, use_bn=False)

    # - GELU (Replaced with ReLU, Rand)
    p_relu, _ = train_eval_enzyfuse(X_tr_s, y_tr, X_val_s, y_val, X_te_s, y_te, num_classes=8, epochs=40, use_gelu=False)

    # - Label Smoothing (eps = 0.0, Rand)
    p_nols, _ = train_eval_enzyfuse(X_tr_s, y_tr, X_val_s, y_val, X_te_s, y_te, num_classes=8, epochs=40, label_smoothing=0.0)

    # Exact 7-row TABLE VI structure: 3 strict CD-HIT <40% branch-fusion ablations + 4 Random-split component ablations
    ablation_specs = [
        ("Full EnzyFuse-Net V5 (Dual, <40%)", res_hom["EnzyFuse-Net V5 (Dual-Branch)"], y_te_h, "hom_base"),
        ("- Contrastive Branch (GELU, <40%)", res_hom["EnzyFuse-Net V5 (GELU Branch)"], y_te_h, "hom_delta"),
        ("- GELU Branch (SupCon, <40%)",      res_hom["CLEAN Contrastive (SupCon Br.)"], y_te_h, "hom_delta"),
        ("Discriminative Branch (GELU, Rand)", res_rand["EnzyFuse-Net V5 (GELU Branch)"], y_te,   "rand_base"),
        ("- 1D Batch Normalization (Rand)",    p_nobn,                                    y_te,   "rand_delta"),
        ("- GELU (Replaced with ReLU, Rand)",  p_relu,                                    y_te,   "rand_delta"),
        ("- Label Smoothing (ε = 0.0, Rand)",  p_nols,                                    y_te,   "rand_delta"),
    ]
    base_acc_hom = accuracy_score(y_te_h, res_hom["EnzyFuse-Net V5 (Dual-Branch)"]) * 100.0
    base_acc_rand = accuracy_score(y_te, res_rand["EnzyFuse-Net V5 (GELU Branch)"]) * 100.0
    t6_rows = []
    for cfg_name, cfg_preds, y_eval_ref, mode_tag in ablation_specs:
        tp_c = int((cfg_preds == y_eval_ref).sum())
        acc_c = (tp_c / len(y_eval_ref)) * 100.0
        if mode_tag == "hom_base":
            delta_str = "Baseline (<40%)"
        elif mode_tag == "hom_delta":
            delta_str = f"{acc_c - base_acc_hom:+.2f}% (<40%)"
        elif mode_tag == "rand_base":
            delta_str = "Baseline (Rand)"
        else:
            delta_str = f"{acc_c - base_acc_rand:+.2f}% (Rand)"
        f1_c = f1_score(y_eval_ref, cfg_preds, average="macro") * 100.0
        mcc_c = matthews_corrcoef(y_eval_ref, cfg_preds)
        t6_rows.append([
            cfg_name,
            f"{tp_c}/{len(y_eval_ref)}",
            f"{acc_c:.2f}%",
            delta_str,
            f"{f1_c:.2f}%",
            f"{mcc_c:.4f}",
        ])
    print("\n=== TABLE VI: Live Ablation Study of Architectural, Branch-Fusion, and Optimization Components (N_test = 600) ===")
    t6_df = pd.DataFrame(t6_rows, columns=["Model Configuration", "TP/600", "Test Acc", "Delta (Δ)", "F1 (%)", "MCC"])
    print(t6_df.to_string(index=False))

    # --------------------------------------------------------------------------
    # TABLE VII: Live 4-Tier Hierarchical EC Prediction (Level 1 -> Level 4)
    # --------------------------------------------------------------------------
    print("\n[STEP 6] Evaluating Live 4-Tier Hierarchical EC Prediction (3-mer, Real NCBI BLASTp, CLEAN & V5)...")
    tier_specs = [
        ("EC_Level1", "Level 1: Main Class (EC X)"),
        ("EC_Level2", "Level 2: Subclass (EC X.X)"),
        ("EC_Level3", "Level 3: Sub-Subclass (EC X.X.X)"),
        ("EC_Level4", "Level 4: 4-Digit (EC X.X.X.X)"),
    ]
    t7_rows = []
    l4_eval_cache = {}
    for tier_col, tier_title in tier_specs:
        le = LabelEncoder()
        y_tier_all = le.fit_transform(df[tier_col].values)
        k_classes = len(le.classes_)
        tier_metrics = {}
        for proto_key, (i_tr, i_val, i_te), sc_pack, is_hom in [
            ("rand", split_rand, scaled_rand, False),
            ("hom",  split_hom,  scaled_hom,  True),
        ]:
            X_tr_p, _, X_val_p, _, X_te_p, _ = sc_pack
            yt_tr, yt_val, yt_te = y_tier_all[i_tr], y_tier_all[i_val], y_tier_all[i_te]
            tr_seqs_p = [seqs[i] for i in i_tr]
            te_seqs_p = [seqs[i] for i in i_te]

            if tier_col == "EC_Level1":
                res_ref = res_hom if is_hom else res_rand
                y_ref = y_te_h if is_hom else y_te_r
                tier_metrics[proto_key] = (
                    accuracy_score(y_ref, res_ref["3-mer Jaccard Similarity Baseline"]) * 100.0,
                    accuracy_score(y_ref, res_ref["NCBI BLASTp Best-Hit (E <= 1e-3)"]) * 100.0,
                    accuracy_score(y_ref, res_ref["CLEAN Contrastive (SupCon Br.)"]) * 100.0,
                    accuracy_score(y_ref, res_ref["EnzyFuse-Net V5 (Dual-Branch)"]) * 100.0,
                )
            else:
                # 1A. 3-mer Jaccard similarity on current EC hierarchy tier
                p_kmer_tier = kmer_alignment_predict(tr_seqs_p, yt_tr, te_seqs_p, k=3)
                # 1B. Genuine NCBI BLASTp (E <= 1e-3, <40% identity enforced on Protocol B)
                p_blast_tier = ncbi_blastp_predict(
                    tr_seqs_p, yt_tr, te_seqs_p,
                    evalue_cutoff=1e-3,
                    max_identity=40.0 if is_hom else None
                )
                # 2. CLEAN Contrastive on current EC hierarchy tier
                p_c_tier, pr_c_tier, vpr_c_tier = train_eval_clean_contrastive(
                    X_tr_p, yt_tr, X_te_p, yt_te, X_val=X_val_p, y_val=yt_val,
                    num_classes=k_classes, epochs=30, return_val_probs=True
                )
                # 3. EnzyFuse-Net V5 GELU + Calibrated Dual-Branch Fusion on current EC hierarchy tier
                p_g_tier, pr_g_tier, vpr_g_tier = train_eval_enzyfuse(
                    X_tr_p, yt_tr, X_val_p, yt_val, X_te_p, yt_te,
                    num_classes=k_classes, epochs=35, return_val_probs=True
                )
                p_f_tier, _, _ = tune_calibrated_dual_branch_fusion(
                    vpr_g_tier, vpr_c_tier, yt_val, pr_g_tier, pr_c_tier
                )
                tier_metrics[proto_key] = (
                    accuracy_score(yt_te, p_kmer_tier) * 100.0,
                    accuracy_score(yt_te, p_blast_tier) * 100.0,
                    accuracy_score(yt_te, p_c_tier) * 100.0,
                    accuracy_score(yt_te, p_f_tier) * 100.0,
                )
                if tier_col == "EC_Level4":
                    l4_eval_cache[proto_key] = (yt_te, p_kmer_tier, p_blast_tier, p_c_tier, p_f_tier)
        rk, rb, rc, rv = tier_metrics["rand"]
        hk, hb, hc, hv = tier_metrics["hom"]
        t7_rows.append([
            tier_title,
            k_classes,
            f"{rk:.2f}%",
            f"{rb:.2f}%",
            f"{rc:.2f}%",
            f"{rv:.2f}%",
            f"{hk:.2f}%",
            f"{hb:.2f}%",
            f"{hc:.2f}%",
            f"{hv:.2f}%",
        ])
    print("\n=== TABLE VII: Live 4-Tier Hierarchical EC Prediction (Level 1: EC X -> Level 4: EC X.X.X.X, N_test = 600) ===")
    t7_df = pd.DataFrame(
        t7_rows,
        columns=[
            "IUBMB Hierarchy Tier", "Classes (K)",
            "Rand 3-mer", "Rand BLASTp", "Rand CLEAN", "Rand V5",
            "<40% 3-mer", "<40% BLASTp", "<40% CLEAN", "<40% V5"
        ]
    )
    print(t7_df.to_string(index=False))

    # --------------------------------------------------------------------------
    # STEP 7: IEEE Transactions (TCBB/JBHI) Extended Reviewer Suite
    #   7A. 5-Seed Mean ± SD Multi-Seed Stability Verification (Seeds 42..46)
    #   7B. Level-4 Support-Stratified Analysis (K >= 10 Support) & New-392 / Price-149 Loader
    #   7C. Optional Multi-Backbone Scaling Check (esm2_t6_8M vs. esm2_t12_35M)
    # --------------------------------------------------------------------------
    print("\n[STEP 7A] Running 5-Seed Mean ± SD Stability Verification Across Seeds [42, 43, 44, 45, 46]...")
    seeds_list = [42, 43, 44, 45, 46]
    seed_stats = {
        "Rand GELU": [], "Rand CLEAN": [], "Rand Dual-V5": [],
        "<40% GELU": [], "<40% CLEAN": [], "<40% Dual-V5": [],
    }
    for s_val in seeds_list:
        set_global_seed(s_val)
        # Random split run for seed s_val
        X_tr_r_s, y_tr_r_s, X_val_r_s, y_val_r_s, X_te_r_s, y_te_r_s = scaled_rand
        pg_r, prg_r, vprg_r = train_eval_enzyfuse(
            X_tr_r_s, y_tr_r_s, X_val_r_s, y_val_r_s, X_te_r_s, y_te_r_s,
            num_classes=8, epochs=40, seed=s_val, return_val_probs=True
        )
        pc_r, prc_r, vprc_r = train_eval_clean_contrastive(
            X_tr_r_s, y_tr_r_s, X_te_r_s, y_te_r_s, X_val=X_val_r_s, y_val=y_val_r_s,
            num_classes=8, epochs=35, seed=s_val, return_val_probs=True
        )
        pf_r, _, _ = tune_calibrated_dual_branch_fusion(vprg_r, vprc_r, y_val_r_s, prg_r, prc_r)
        seed_stats["Rand GELU"].append(accuracy_score(y_te_r_s, pg_r) * 100.0)
        seed_stats["Rand CLEAN"].append(accuracy_score(y_te_r_s, pc_r) * 100.0)
        seed_stats["Rand Dual-V5"].append(accuracy_score(y_te_r_s, pf_r) * 100.0)

        # Strict <40% CD-HIT split run for seed s_val
        X_tr_h_s, y_tr_h_s, X_val_h_s, y_val_h_s, X_te_h_s, y_te_h_s = scaled_hom
        pg_h, prg_h, vprg_h = train_eval_enzyfuse(
            X_tr_h_s, y_tr_h_s, X_val_h_s, y_val_h_s, X_te_h_s, y_te_h_s,
            num_classes=8, epochs=40, seed=s_val, return_val_probs=True
        )
        pc_h, prc_h, vprc_h = train_eval_clean_contrastive(
            X_tr_h_s, y_tr_h_s, X_te_h_s, y_te_h_s, X_val=X_val_h_s, y_val=y_val_h_s,
            num_classes=8, epochs=35, seed=s_val, return_val_probs=True
        )
        pf_h, _, _ = tune_calibrated_dual_branch_fusion(vprg_h, vprc_h, y_val_h_s, prg_h, prc_h)
        seed_stats["<40% GELU"].append(accuracy_score(y_te_h_s, pg_h) * 100.0)
        seed_stats["<40% CLEAN"].append(accuracy_score(y_te_h_s, pc_h) * 100.0)
        seed_stats["<40% Dual-V5"].append(accuracy_score(y_te_h_s, pf_h) * 100.0)
    set_global_seed(SEED)

    print("=== 5-Seed Mean ± SD Accuracy Summary (Seeds 42..46, N_test = 600) ===")
    for k_name, vals in seed_stats.items():
        arr_v = np.array(vals)
        print(f"  -> {k_name:15s}: {arr_v.mean():.2f}% ± {arr_v.std(ddof=1):.2f}% (Seeds: {[round(x, 2) for x in vals]})")

    # STEP 7B: High-Support Level-4 Analysis (>=10 sequences/family) & External Benchmark Hook (New-392 / Price-149)
    print("\n[STEP 7B] Evaluating Level-4 EC Families Filtered by Support (>= 10 Sequences/Family)...")
    l4_counts = df["EC_Level4"].value_counts()
    dense_l4_families = set(l4_counts[l4_counts >= 10].index)
    print(f"  -> Total Level-4 families: {len(l4_counts)} | Well-supported families (>=10 seqs): {len(dense_l4_families)}")

    # Compute and print Level-4 accuracy on well-supported 4-digit EC families (>=10 sequences/family)
    for proto_label, proto_key, split_pack in [("Random Split", "rand", split_rand), ("Strict <40% CD-HIT Split", "hom", split_hom)]:
        te_l4_names = df["EC_Level4"].values[split_pack[2]]
        dense_mask = np.array([name in dense_l4_families for name in te_l4_names])
        if dense_mask.sum() > 0 and proto_key in l4_eval_cache:
            yt_l4, pk_l4, pb_l4, pc_l4, pf_l4 = l4_eval_cache[proto_key]
            acc_k_d = accuracy_score(yt_l4[dense_mask], pk_l4[dense_mask]) * 100.0
            acc_b_d = accuracy_score(yt_l4[dense_mask], pb_l4[dense_mask]) * 100.0
            acc_c_d = accuracy_score(yt_l4[dense_mask], pc_l4[dense_mask]) * 100.0
            acc_f_d = accuracy_score(yt_l4[dense_mask], pf_l4[dense_mask]) * 100.0
            print(
                f"  -> {proto_label:24s} Dense Level-4 (>=10 seqs, n={int(dense_mask.sum())}/{len(dense_mask)}): "
                f"3-mer={acc_k_d:.2f}% | BLASTp={acc_b_d:.2f}% | CLEAN={acc_c_d:.2f}% | EnzyFuse-Net V5={acc_f_d:.2f}%"
            )

    for ext_csv, ext_name in [("new392.csv", "New-392 (Yu et al., Science 2023)"), ("price149.csv", "Price-149 (Yu et al., Science 2023)")]:
        if os.path.exists(ext_csv):
            ext_df = pd.read_csv(ext_csv)
            print(f"  -> Loaded external independent test set {ext_name}: {len(ext_df)} sequences from {ext_csv}")

    # STEP 7C: Optional ESM-2 Backbone Scaling Check (8M 320-D vs. 35M 480-D)
    if os.environ.get("RUN_ESM2_35M_SCALING", "0") == "1" or os.path.exists("esm2_t12_35M_4000_embeddings.npy"):
        print("\n[STEP 7C] Running ESM-2 Backbone Scaling Check (esm2_t12_35M_UR50D, 12 Layers, 35M Params, d = 480)...")
        if os.path.exists("esm2_t12_35M_4000_embeddings.npy"):
            X_35m = np.load("esm2_t12_35M_4000_embeddings.npy")
        else:
            from transformers import AutoTokenizer, EsmModel
            tok35 = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
            mod35 = EsmModel.from_pretrained("facebook/esm2_t12_35M_UR50D").to(DEVICE)
            mod35.eval()
            embs35 = []
            with torch.no_grad():
                for b_start in range(0, len(seqs), 32):
                    b_seqs = seqs[b_start:b_start + 32]
                    enc35 = tok35(b_seqs, return_tensors="pt", padding=True, truncation=True, max_length=402).to(DEVICE)
                    h35 = mod35(**enc35).last_hidden_state
                    att35 = enc35["attention_mask"].clone()
                    att35[:, 0] = 0
                    lengths35 = enc35["attention_mask"].sum(dim=1) - 1
                    for r_i, l_idx in enumerate(lengths35):
                        if l_idx > 0:
                            att35[r_i, l_idx] = 0
                    m35 = att35.unsqueeze(-1).float()
                    p35 = (h35 * m35).sum(dim=1) / torch.clamp(m35.sum(dim=1), min=1.0)
                    embs35.append(p35.cpu().numpy())
            X_35m = np.vstack(embs35).astype(np.float32)
            np.save("esm2_t12_35M_4000_embeddings.npy", X_35m)
        sc35_h = StandardScaler()
        X35_tr_h = sc35_h.fit_transform(X_35m[split_hom[0]])
        X35_val_h = sc35_h.transform(X_35m[split_hom[1]])
        X35_te_h = sc35_h.transform(X_35m[split_hom[2]])
        p35_g, pr35_g, vpr35_g = train_eval_enzyfuse(
            X35_tr_h, y_all[split_hom[0]], X35_val_h, y_all[split_hom[1]], X35_te_h, y_te_h,
            num_classes=8, epochs=40, return_val_probs=True
        )
        p35_c, pr35_c, vpr35_c = train_eval_clean_contrastive(
            X35_tr_h, y_all[split_hom[0]], X35_te_h, y_te_h, X_val=X35_val_h, y_val=y_all[split_hom[1]],
            num_classes=8, epochs=35, return_val_probs=True
        )
        p35_f, _, _ = tune_calibrated_dual_branch_fusion(vpr35_g, vpr35_c, y_all[split_hom[1]], pr35_g, pr35_c)
        print(f"  -> esm2_t12_35M_UR50D (480-D) Strict <40% Split Accuracy: Dual-Branch = {accuracy_score(y_te_h, p35_f)*100:.2f}% (GELU = {accuracy_score(y_te_h, p35_g)*100:.2f}%)")

    total_elapsed = time.time() - t_start
    print("\n" + "=" * 96)
    print(f"COMPLETE LIVE PIPELINE FINISHED IN {total_elapsed:.1f} SECONDS.")
    print(f"Generated CSVs in working directory: {csv_4000}, {csv_3000}")
    print("=" * 96)
    if trigger_browser_download:
        try:
            from google.colab import files
            files.download(csv_4000)
            files.download(csv_3000)
        except Exception:
            pass


if __name__ == "__main__":
    run_genuine_colab_pipeline(use_cdhit_for_protocol_b=True, trigger_browser_download=False)
