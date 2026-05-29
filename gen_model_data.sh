output_dir=./output_001
n_samples=5
context_len_interval=2
python gen_model_data_fast.py --output_dir $output_dir --n_samples $n_samples --use_pool --context_len_min 30 --context_len_max 40 --context_len_interval $context_len_interval &
python gen_model_data_fast.py --output_dir $output_dir --n_samples $n_samples --use_pool --context_len_min 170 --context_len_max 180 --context_len_interval $context_len_interval &

python gen_model_data_fast.py --output_dir $output_dir --n_samples $n_samples --use_pool --context_len_min 50 --context_len_max 51 --flag val &
python gen_model_data_fast.py --output_dir $output_dir --n_samples $n_samples --use_pool --context_len_min 50 --context_len_max 51 --flag test &
