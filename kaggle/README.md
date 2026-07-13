# Kaggle Canonical Smoke Package

This package prepares and validates SemKey runs without placing ZuCo data in Git.

## Required repository state

The notebook must clone a user-controlled repository/fork containing the canonical adapter changes and checkout an exact commit. The current upstream `xmed-lab/SemKey` commit does not contain these local changes. Never run from a floating branch for paper evidence.

## Private input datasets

Use two versions:

1. **Source dataset** (initial upload only)
   - `zuco_eeg_label_8variants.df`
   - `canonical_validation_manifest.csv`
   - `source_checksums.json`
2. **Derived sharded dataset** (normal training/smoke input)
   - `shards/shard_00000.npz`, ...
   - `shards/index.csv`
   - `metadata/shard_manifest.json`
   - `manifests/canonical_full_manifest.csv`
   - `metadata/canonical_full_contract_report.json`

Keep both private unless upstream redistribution terms explicitly permit publication. Git contains only code, schemas, and example manifests.

Because the project repository is private, add a private Kaggle Secret named
`GITHUB_TOKEN` containing a fine-grained GitHub token with read-only Contents
access to `thestonedape/task-aware-eeg2text`. The notebooks use `GIT_ASKPASS`;
the token is not embedded in notebook source or printed to output.

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
4. For the source upload, run canonical schema-v2 `prepare_shards.py` once and require exact agreement with the frozen 2,200-row validation manifest.
5. Attach the derived dataset and run a real `--batch-size 1` shard smoke.
6. Run the SemKey-compatible loader and GLIM representation adapter on the same ordered real sample ID.
7. Run SemKey imports and scheduler/prompt/identity smokes.
8. Freeze the evaluation manifests and run the feature-admission controls before any pilot training.

## Source-to-shard conversion

After the private source dataset is fully processed by Kaggle, do not upload it
again. Attach it to a high-RAM Kaggle session and run
`convert_source_to_shards.ipynb`. The notebook verifies the frozen source and
manifest hashes, creates row-addressable shards containing the complete raw and
canonical task metadata, writes the full 22,335-row canonical manifest, and runs
both the storage smoke and trainable shard-backed loader smoke. Save its output
directory as a second private Kaggle dataset. Normal
training notebooks attach that derived dataset instead of deserializing the
14+ GB source pickle.

The source pickle is over 14 GB and pandas must deserialize it as a whole. Conversion therefore belongs in a high-RAM Kaggle session, not the local desktop smoke.

The frozen GLIM dataframe does **not** contain SemKey-generated binary sentiment,
topic, or GPT-2 surprisal fields. Schema v2 deliberately leaves their availability
masks false instead of fabricating them from native labels. Generate and version
those text-derived compatibility targets later from the small shard index; the
14+ GB EEG dataframe must not be deserialized again for label enrichment.

## Balanced GLIM/SemKey representation gate

GLIM is the primary representation/alignment and direct-retrieval candidate;
SemKey supplies the richer-feature-head and guided-generation interfaces. They
are joined through project-owned adapters rather than trained as two full
pipelines. Attach the official Figshare GLIM checkpoint as a private Kaggle
input and verify SHA-256
`25fcd31d1d6cafc9a0656c50a4916ba6ee106884b269d347284784cc0522c8ba`. After
attaching the derived shards and checkpoint,
run:

```powershell
python kaggle/smoke_glim_sharded_representation.py `
  --dataset-root <derived-dataset-root> `
  --glim-root <pinned-glim-checkout> `
  --checkpoint <glim-checkpoint> `
  --phase val --prompt-mode canonical
```

The canonical adapter maps the released GLIM prompt as the checkpoint-compatible
base and adds separate zero-initialized trainable SR/NR/TSR deltas. The smoke
must report the same `sample_id` and source dataframe row as the SemKey-compatible
loader. Similar module names across GLIM and SemKey are not checkpoint evidence.
