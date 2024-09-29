source .venv/bin/activate

python3 learn_shapes.py /mnt/raid0b/shapes/ logs_scale \
  --net=ann --n_scales=1 --n_angles=1 --n_ratios=1 --timesteps=20 \
  --dataset_filter="-s" --coordinate=dsnt \
  --max_epochs=100 --accelerator=gpu --devices=1