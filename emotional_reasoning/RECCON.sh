CUDA_VISIBLE_DEVICES='7' python main.py \
    --DATASET RECCON \
    --model_checkpoint roberta-large \
    --alpha 0.8 \
    --NUM_TRAIN_EPOCHS 10 \
    --BATCH_SIZE 1 \
    --model_save_dir ./model_save_dir/RECCON \
    --mode train \
    --LR 1e-5 \
    --SEED 42 \
    --utt_kl_weight 0.005
    --ROOT_DIR ./comet_enhanced_data/ \
    --CONV_NAME multidim_hgt \
    --COMET_HIDDEN_SIZE 768 \
    --CUDA
    --experiment 10 \