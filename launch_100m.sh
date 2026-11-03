#!/bin/bash
set -e

cd /home/duoneural/srsn_ryzen
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PATH="/home/duoneural/srsn_env/bin:$PATH"

# Ensure RAM-disk datasets are hot in /dev/shm
if [ ! -f /dev/shm/train_golden.bin ]; then
    echo "[Setup] Copying train_golden.bin to /dev/shm..."
    cp /home/duoneural/srsn_ryzen/train_golden.bin /dev/shm/train_golden.bin
fi

if [ ! -f /dev/shm/val_golden.bin ]; then
    echo "[Setup] Copying val_golden.bin to /dev/shm..."
    cp /home/duoneural/srsn_ryzen/val_golden.bin /dev/shm/val_golden.bin
fi

echo "[Launch] Starting SRSN 100M-Token Extended Pretraining Daemon (Step 325 -> 3250)..."
nohup taskset -c 0,1,2,3 /home/duoneural/srsn_env/bin/python -u /home/duoneural/srsn_ryzen/train_srsn.py \
  --train-bin /dev/shm/train_golden.bin \
  --val-bin /dev/shm/val_golden.bin \
  --resume-checkpoint /home/duoneural/srsn_ryzen/checkpoints/srsn_step0325.pt \
  --total-optimizer-steps 3250 \
  --warmup-steps 5 \
  --resume-warmup-steps 25 \
  --save-interval 25 \
  --eval-interval 25 > /home/duoneural/srsn_ryzen/train_100m.log 2>&1 &

PID=$!
echo $PID > /home/duoneural/srsn_ryzen/train_100m.pid
echo "[Launch] SRSN 100M Daemon started with PID $PID on CPU cores 0,1,2,3."
