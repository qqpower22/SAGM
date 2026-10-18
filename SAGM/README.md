# SAGM

SAGM（Structure-Aware Grid Model）

## train case

```bash
python main_repro.py \
  --config ./config/cadec.json \
  --device 0 \
  --vib_beta 0.12 \
  --seed 123 \
  --use_contrastive \
  --negative_strategy replace,extend,delete,shuffle \
  --session_name cadec_full \
  --dev_only
```

## test case

```bash
python main_repro.py \
  --config ./config/cadec.json \
  --device 0 \
  --vib_beta 0.12 \
  --seed 123 \
  --use_contrastive \
  --negative_strategy replace,extend,delete,shuffle \
  --session_name cadec_full \
  --test_only
```
