# Kaggle Canonical Smoke Package

This package prepares and validates SemKey runs without placing ZuCo data in Git.

## Required repository state

The notebook must clone a user-controlled repository/fork containing the canonical adapter changes and checkout an exact commit. The current upstream `xmed-lab/SemKey` commit does not contain these local changes. Never run from a floating branch for paper evidence.

## Private input datasets

Use two versions:

1. **Source dataset** (initial upload only)
   - `source/zuco_eeg_label_8variants.df`
   - `manifests/canonical_validation_manifest.csv`
   - `metadata/source_dataset.json`
2. **Derived sharded dataset** (normal training/smoke input)
   - `shards/shard_00000.npz`, ...
   - `shards/index.csv`
   - `metadata/shard_manifest.json`
   - the canonical manifest and contract report

Keep both private unless upstream redistribution terms explicitly permit publication. Git contains only code, schemas, and example manifests.

## Upload the private source dataset

From the workspace root, run this in PowerShell on the faster connection:

```powershell
powershell -ExecutionPolicy Bypass -File .\SemKey\kaggle\upload_source_dataset.ps1 -StopExistingUpload
```

The switch stops the previously recorded slow upload before creating a fresh
dataset version. Omit it when no earlier upload process is running. The script
reads the Kaggle credential from `KAGGLE_API_TOKEN` or
`~/.kaggle/access_token`; the token is never embedded in the repository.

## Gate order

1. Clone and verify the explicit commit.
2. Discover the attached dataset below `/kaggle/input`.
3. Run `smoke_input.py --metadata-only`.
4. For a source upload, run `prepare_shards.py` once and publish its `/kaggle/working` output as a new private Kaggle Dataset version.
5. Attach the derived dataset and run a real `--batch-size 1` shard smoke.
6. Run SemKey imports and scheduler/prompt/identity smokes.
7. Only after these pass, load a base model/checkpoint for a forward-only smoke.

## Source-to-shard conversion

After the private source dataset is fully processed by Kaggle, do not upload it
again. Attach it to a high-RAM Kaggle session and run
`convert_source_to_shards.ipynb`. The notebook verifies the frozen source and
manifest hashes, creates row-addressable shards containing all SemKey Stage-1
metadata, and runs both the storage smoke and the trainable shard-backed loader
smoke. Save its output directory as a second private Kaggle dataset. Normal
training notebooks attach that derived dataset instead of deserializing the
14+ GB source pickle.

The source pickle is over 14 GB and pandas must deserialize it as a whole. Conversion therefore belongs in a high-RAM Kaggle session, not the local desktop smoke.
