GPU_ID=${1}

INIT_LR=1e-6
EPOCH=100
BATCH_SIZE=6
INPUT_CHANNEL=3

LR_RECONSTRUCTION=1e-3
LOSS_WEIGHT=1

RESUME='./pretrained_models/pixel_pggan_celeba.pth.tar'
SR_RESUME='./pretrained_models/sr_vgg_epoch_last.pth'

OUTPUT_PATH='./output_end2end/pixel_pggan_colorization'

DATA_ROOT_POS='./data/CelebA'
DATA_ROOT_NEG='./data/CelebA_ProGAN'


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
  --idx-stages 0 \
  --mode-sr colorization

