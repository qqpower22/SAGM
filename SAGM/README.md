# SAGM

SAGM（Structure-Aware Grid Model）

## train case

```bash
python main_repro.py \
  --config ./config/cadec.json \
  --device 0 \
  --vib_beta 0.12 \
  --seed 123 \
  --use_vib \
  --use_contrastive \
  --use_entropy_fusion 1 \
  --session_name cadec_A \
  --dev_only
```

