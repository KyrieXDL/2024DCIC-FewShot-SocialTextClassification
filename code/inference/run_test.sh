CUDA_VISIBLE_DEVICES=0 python main_retrieval.py \
--data_path '../raw_data/test_data.txt' \
--batch_size 16 \
--task_name 'base_retrieval' \
--encoder_dir '../user_data/m3e-large' \
--max_len 512 \
--phase 'test' \

