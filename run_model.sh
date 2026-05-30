data_path=./output_001
e_layer=6
d_model=512
d_ff=2048
model_id=output_001_el${e_layer}_dm${d_model}_dff${d_ff}

python run.py --task_name tsicl --is_training 1 --root_path ${data_path} \
--model_id ${model_id} --model tsicl --data tsicl --seq_len 512 --label_len 0 --pred_len 512 \
--e_layers ${e_layer} --d_layers 1 --factor 3 --des 'Exp' --d_model $d_model --d_ff $d_ff --itr 1 --batch_size 100 \
--num_workers 1 --train_epochs 50 --patience 5 --learning_rate 0.00005 --lradj step \
> run_logs/${model_id}.log
