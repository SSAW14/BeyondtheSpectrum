# example to test stage5 based detectors in CelebA dataset

INPUT_CHANNEL=512
GPU_ID=${1}

DATA_ROOT_POS='./data/CelebA'
DATA_ROOT_NEG='./data/CelebA_ProGAN'

RESUME='./pretrained_models/stylegan_celeba_stage5_noising/cls.pth.tar'
SR_RESUME='./pretrained_models/stylegan_celeba_stage5_noising/sr.pth.tar'
SAVE_PATH='./output/train_stylegan_stage5_noising_test_pggan'

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
  --idx-stages 5
