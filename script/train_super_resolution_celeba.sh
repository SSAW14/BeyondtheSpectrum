# script to train super resolution model

GPU_ID=${1}

python train_sr.py --train-file "./data/celeba_example_train.txt" \
                --eval-file "./celeba_example_val.txt" \
                --outputs-dir "./output_hr" \
                --scale 4 \
                --num-features 64 \
                --growth-rate 64 \
                --num-blocks 16 \
                --num-layers 8 \
                --lr 1e-4 \
                --lr-decay-epoch 225 \
                --batch-size 12 \
                --patch-size 24 \
                --num-epochs 750 \
                --num-save 15 \
                --num-workers 8 \
                --gpu-id ${GPU_ID}
