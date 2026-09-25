# Google Colab T4 GPU Execution Guide

This guide details how to run the entire **Amazon ML Challenge 2026: Business Entity Resolution** pipeline on Google Colab with an **NVIDIA Tesla T4 GPU**.

---

## 1. Prepare Archive Locally

In your local terminal, run:
```bash
make collab-prep
```
This builds `aml_collab.zip` containing the complete codebase, configuration, validators, and datasets.

---

## 2. Set Up Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Click **New Notebook**.
3. Enable GPU Acceleration:
   - Navigate to `Runtime` -> `Change runtime type`
   - Under **Hardware accelerator**, select **T4 GPU**
   - Click **Save**.

---

## 3. Upload `aml_collab.zip`

In the left sidebar of Colab:
1. Click the **Files** icon (folder icon).
2. Click the **Upload to session storage** button.
3. Select `aml_collab.zip` from your computer (wait for the upload progress circle to complete).

*(Alternative: If you uploaded `aml_collab.zip` to Google Drive, see Section 5 below).*

---

## 4. Run the Pipeline (Single Copy-Paste Cell)

Copy and paste the entire block below into a code cell in Google Colab and run it (`Shift + Enter`):

```python
# ======================================================================
# Amazon ML Challenge 2026: End-to-End Execution Cell
# ======================================================================

# 1. Verify NVIDIA T4 GPU
!nvidia-smi

# 2. Unpack the uploaded codebase and dataset
print("\n[Colab Setup] Extracting aml_collab.zip...")
!unzip -q -o /content/aml_collab.zip -d /content/aml
%cd /content/aml

# 3. Install dependencies
print("\n[Colab Setup] Installing dependencies...")
!pip install -q -r chimera_submission/code/business_entity_resolution/requirements.txt

# 4. Run End-to-End Pipeline with GPU Acceleration & Direct TSV Streaming
print("\n[Colab Execution] Running Business Entity Resolution Pipeline...")
!PYTHONPATH=. python3 chimera_submission/code/business_entity_resolution/src/main.py \
    --train-dir dataset/student_resource/dataset/train \
    --test-dir dataset/student_resource/dataset/test \
    --output-dir chimera_submission/output \
    --device auto \
    --n-jobs -1 \
    --max-train-queries 40000

# 5. Run Official Submission Validation
print("\n[Colab Validation] Running Official Competition Validator...")
!python3 utils/validate_submission.py \
    --matching chimera_submission/output/matching_results.tsv \
    --candidate chimera_submission/output/candidate_pairs.tsv \
    --test-dir dataset/student_resource/dataset/test

# 6. Package and Automatically Trigger Download of Submission Results
print("\n[Colab Packaging] Packaging final submission files...")
!zip -q -j /content/submission_results.zip \
    chimera_submission/output/matching_results.tsv \
    chimera_submission/output/candidate_pairs.tsv

from google.colab import files
files.download('/content/submission_results.zip')
print("\n Done! submission_results.zip is downloading to your computer.")
```

---

## 5. Alternative: Running via Google Drive (For Large File Resumes)

If you prefer uploading `aml_collab.zip` to Google Drive rather than session storage:

```python
from google.colab import drive
drive.mount('/content/drive')

# Replace with the path to aml_collab.zip in your Google Drive:
!cp /content/drive/MyDrive/aml_collab.zip /content/aml_collab.zip
!unzip -q -o /content/aml_collab.zip -d /content/aml
%cd /content/aml

!pip install -q -r chimera_submission/code/business_entity_resolution/requirements.txt

!PYTHONPATH=. python3 chimera_submission/code/business_entity_resolution/src/main.py \
    --train-dir dataset/student_resource/dataset/train \
    --test-dir dataset/student_resource/dataset/test \
    --output-dir chimera_submission/output \
    --device auto \
    --n-jobs -1 \
    --max-train-queries 40000
```
