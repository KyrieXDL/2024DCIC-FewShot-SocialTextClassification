CUDA_VISIBLE_DEVICES=0 python main_pretrain.py \
--data_path '../raw_data/train_data.txt' \
--batch_size 8 \
--task_name 'models' \
--encoder_dir '../user_data/m3e-large' \
--max_len 512 \
--random_lr 1e-4 \
--pretrained_lr 1e-5 \
--schedule_type 'none' \
--use_fp16 \
--valid_ratio 0. \
--epochs 2 \
--preprocess \
--use_fgm \
--output_dir '../user_data' \



