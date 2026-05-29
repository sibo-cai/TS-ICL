output_dir=./output_001
n_samples=5000
context_len_interval=2
context_min_list=(30 40 50 60 70 80)
for i_i in "${context_min_list[@]}"
do
  echo "python gen_model_data_fast.py --output_dir $output_dir --n_samples $n_samples --use_pool --context_len_min $i_i --context_len_max $((i_i+10)) --context_len_interval $context_len_interval &"
done
