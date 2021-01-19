cd /home/MLTL-INTERN/zhangzs/zhangzs/Dialogue/molweni-eval/mtl_electra/mrc_electra
### pre-work ###
export PROJECT_DIR=./
### GPU parts start : Ubuntu###
export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR
export CUDA_VISIBLE_DEVICES=`perl /share03/zhangzs/idle-gpus.pl -n 1`
source activate cmrc
### GPU parts ends ###

python run_molweni_mdfn.py \
--model_type electra-molweni-baseline \
--model_name_or_path "bert-base-uncased" \
--data_dir "molweni/Molweni/MRC" \
--version_2_with_negative \
--output_dir "experiment/multi_ep3_lr5e-5_dp2" \
--do_train \
--do_eval \
--do_lower_case \
--overwrite_output_dir \
--num_train_epochs 3 \
--per_gpu_train_batch_size 8 \
--learning_rate 5e-5 \
--save_steps 500 \
--dropout 0.2 \
--eval_all_checkpoints \
--task 0

python run_molweni_mdfn.py \
--model_type electra-molweni-baseline \
--model_name_or_path "bert-base-uncased" \
--data_dir "molweni/Molweni/MRC" \
--version_2_with_negative \
--output_dir "experiment/multi_ep4_lr5e-5_dp2" \
--do_train \
--do_eval \
--do_lower_case \
--overwrite_output_dir \
--num_train_epochs 4 \
--per_gpu_train_batch_size 8 \
--learning_rate 5e-5 \
--save_steps 500 \
--dropout 0.2 \
--eval_all_checkpoints \
--task 0

python run_molweni_mdfn.py \
--model_type electra-molweni-baseline \
--model_name_or_path "bert-base-uncased" \
--data_dir "molweni/Molweni/MRC" \
--version_2_with_negative \
--output_dir "experiment/multi_ep2_lr5e-5_dp2_mrc" \
--do_train \
--do_eval \
--do_lower_case \
--overwrite_output_dir \
--num_train_epochs 2 \
--per_gpu_train_batch_size 8 \
--learning_rate 5e-5 \
--save_steps 500 \
--dropout 0.2 \
--eval_all_checkpoints \
--task 1

python run_molweni_mdfn.py \
--model_type electra-molweni-baseline \
--model_name_or_path "bert-base-uncased" \
--data_dir "molweni/Molweni/MRC" \
--version_2_with_negative \
--output_dir "experiment/multi_ep2_lr5e-5_dp2_dp" \
--do_train \
--do_eval \
--do_lower_case \
--overwrite_output_dir \
--num_train_epochs 2 \
--per_gpu_train_batch_size 8 \
--learning_rate 5e-5 \
--save_steps 500 \
--dropout 0.2 \
--eval_all_checkpoints \
--task 2
