# example to test stage5 based detectors in CelebA dataset

INPUT_CHANNEL=3
GPU_ID=${1}

DATA_ROOT_POS='./data/CelebA'
DATA_ROOT_NEG='./data/CelebA_StyleGAN'

RESUME='./pretrained_models/pixel_pggan_celeba.pth.tar'
SR_RESUME='./pretrained_models/sr_vgg_epoch_last.pth'
SAVE_PATH='./output/train_pggan_pixel_test_stylegan'

python test.py -a resnet50 \
  --gpu ${GPU_ID} \
  --data-root-pos ${DATA_ROOT_POS} \
  --data-root-neg ${DATA_ROOT_NEG} \
  --input-channel ${INPUT_CHANNEL} \
  --resume ${RESUME} \
  --sr-weights-file ${SR_RESUME} \
  --save_path ${SAVE_PATH} \
  --no_dilation \
  --sr-scale 4 \
  --sr-num-features 64 \
  --sr-growth-rate 64 \
  --sr-num-blocks 16 \
  --sr-num-layers 8 \
  --idx-stages 0
