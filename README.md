# MST-PL

Code accompanying our journal manuscript, **Communication-Efficient Distributed Domain Generalization for Resource-Constrained Edge Devices via Unsupervised Prompt Learning**, submitted to **IEEE Transactions on Parallel and Distributed Systems (TPDS)**.

The repository builds on FaST-PT and includes the DPL unsupervised extension described below.

## Environment
Configure the environment according to environment.yml

## Multimodal Style Transfer
```
python MST_dataset.py --config configs/dataset/lads_source_domain.yaml EXP.N_CTX_GLOBAL=length EXP.N_CTX_DOMAIN=length EXP.SEED=seed
```

## Prompt Training with all data
```
python main.py --config configs/DomainNet/FASTPT_target_domain.yaml
```

## Prompt Training under few-shot scenarios
```
python main_shot.py --config configs/DomainNet/FASTPT_target_domain.yaml EXP.SHOT=shot_number
```

## Unsupervised extension: DPL

The additional unsupervised runner implements pseudo-label initial filtering (PLIF), purification (PLP), and expansion (PLE). Its default configuration in `configs/dpl_protocol.json` uses 20 rounds, five local epochs per global/domain stage, and 50 MST epochs. The existing FaST-PT files and directory layout are retained.

Use Python 3.11 and install the DPL dependencies:
```bash
python -m pip install numpy Pillow torch torchvision git+https://github.com/openai/CLIP.git
```

Place PACS images under `DATA/pacs/images/<domain>/<class>/`. Replace `DATA`, `WORK`, and `CHECKPOINT` below with your dataset root, work directory, and official OpenAI CLIP ViT-L/14 checkpoint.

```bash
# Create a PACS membership manifest.
python helpers/dpl_manifest.py --dataset PACS --target-domain sketch \
  --data-root DATA --output WORK/PACS_sketch.json --seed 0

# Extract source-training weak/strong features with frozen CLIP.
python -m helpers.dpl_data --manifest WORK/PACS_sketch.json \
  --clip-checkpoint CHECKPOINT --output WORK/features --device cuda:0

# Run full DPL, fixed-initial labels, and expansion-only policies.
python main_un.py --training-manifest WORK/features/training_manifest.json \
  --clip-checkpoint CHECKPOINT --output WORK/training --device cuda:0

# Evaluate source pseudo-label traces after training.
for dpl_policy in full initial_fixed expansion_only; do
  python evaluate_dpl.py --trace "WORK/training/$dpl_policy/trace" \
    --ground-truth WORK/features/evaluation_only/ground_truth.npz \
    --output "WORK/evaluation/$dpl_policy"
done
```

Use new or empty output directories. Ground truth is separated from training inputs and used only for offline diagnostics. Source-validation and held-out target images do not enter feature preparation or training.
