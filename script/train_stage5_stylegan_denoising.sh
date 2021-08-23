GPU_ID=${1}

INIT_LR=1e-6
EPOCH=100
BATCH_SIZE=6
INPUT_CHANNEL=512

LR_RECONSTRUCTION=1e-3
LOSS_WEIGHT=1

RESUME='./pretrained_models/stylegan_celeba_stage5_noising/cls.pth.tar'
SR_RESUME='./pretrained_models/stylegan_celeba_stage5_noising/sr.pth.tar'

OUTPUT_PATH='./output_end2end/stage5_stylegan_noising'

DATA_ROOT_POS='./data/CelebA'
DATA_ROOT_NEG='./data/CelebA_StyleGAN'


python train.py -a resnet50 \
  --gpu ${GPU_ID} \
  --batch-size ${BATCH_SIZE} \
  --lr ${INIT_LR} \
  --epochs ${EPOCH} \
  --data-root-pos ${DATA_ROOT_POS} \
  --data-root-neg ${DATA_ROOT_NEG} \
  --input-channel ${INPUT_CHANNEL} \
  --resume ${RESUME} \
  --sr-weights-file ${SR_RESUME} \
  --output-path ${OUTPUT_PATH} \
  --lr-sr ${LR_RECONSTRUCTION} \
  --lw-sr ${LOSS_WEIGHT} \
  --no_dilation \
  --sr-scale 4 \
  --sr-num-features 64 \
  --sr-growth-rate 64 \
  --sr-num-blocks 16 \
  --sr-num-layers 8 \
  --idx-stages 5 \
  --mode-sr denoising

