MODEL=CLIP_B16
OUTPUT_DIR=./outputs_sun
DATA_PATH=/home/prp2/zzn/data/sun397
DATASETNAME=sun397

echo $OUTPUT_DIR
mkdir -p $OUTPUT_DIR
cp $0 $OUTPUT_DIR
currentTime=$(date +"%Y_%m_%d_%H_%M_%S")

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,2,1,3 python -m torch.distributed.launch --nproc_per_node=2 run_class_finetuning.py \
    --dict_path "/home/prp2/zzn/clip/FS-SD-CLIP/data/sun397.json" \
    --model ${MODEL} \
    --is_sun \
    --model_sd "LINEAR_HEAD" \
    --sd \
    --is_fewshot \
    --shot 8 \
    --warmup_epochs 5 --epochs 180 \
    --data_path $DATA_PATH \
    --drop 0.4 \
    --attn_drop_rate 0.2 \
    --input_size 224 \
    --finetune True \
    --dataset_name $DATASETNAME \
    --num_workers 8 \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 32 --lr 1e-4 --update_freq 1 \
    --layer_decay 0.7 \
    --drop_path 0 \
    --dist_eval --eval_all \
    --clip_mean_and_std \
    --layer_scale_init_value 0 \
    --abs_pos_emb --disable_rel_pos_bias \
    --weight_decay 0.05 --mixup 0 --cutmix 0 \
    --nb_classes 1000  --model_prefix visual.\
    --model_ema --model_ema_decay 0.9998 \
    --save_ckpt_freq 120 \
    --no_save_ckpt \
    --time ${currentTime} \
    2>&1 | tee -a ${OUTPUT_DIR}/logs/log_${currentTime}.txt
