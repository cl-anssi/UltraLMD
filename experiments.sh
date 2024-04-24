lanl=/path/to/lanl/dataset/
optc=/path/to/optc/dataset/ecar/
python script/preprocessing.py --lanl_dir $lanl --optc_dir $optc --output_dir datasets/

device=cuda
# Zero-shot experiments
for ckpt in $(pwd)/ckpts/*; do
	for dataset in lanl lanl-rich optc optc-rich; do
		python script/eval_ultra.py --data_dir datasets/$dataset/raw/ --ckpt $ckpt --output_dir results/$dataset/ --device $device;
	done
done

# Fine-tuning
for ckpt in $(pwd)/ckpts/*; do
	for dataset in LANL LANLRich OpTC OpTCRich; do
		for seed in 0 1 2 3 4 5 6 7 8 9; do
			python script/run.py -c config/transductive/finetuning.yaml --ckpt $ckpt --seed $seed --dataset $dataset --gpus [0] --root $(pwd)/datasets/ --output_dir $(pwd)/output_ft/;
		done
	done
done

# Evaluation with fine-tuning on same dataset
for log in output_ft/Ultra/LANL/*/log.txt; do
	python script/eval_ultra.py --data_dir datasets/lanl/raw/ --log_file $log --output_dir results/lanl/ --device $device;
done
for log in output_ft/Ultra/LANLRich/*/log.txt; do
	python script/eval_ultra.py --data_dir datasets/lanl-rich/raw/ --log_file $log --output_dir results/lanl-rich/ --device $device;
done
for log in output_ft/Ultra/OpTC/*/log.txt; do
	python script/eval_ultra.py --data_dir datasets/optc/raw/ --log_file $log --output_dir results/optc/ --device $device;
done
for log in output_ft/Ultra/OpTCRich/*/log.txt; do
	python script/eval_ultra.py --data_dir datasets/optc-rich/raw/ --log_file $log --output_dir results/optc-rich/ --device $device;
done

# Evaluation with fine-tuning on other dataset

for log in output_ft/Ultra/OpTC/*/log.txt; do
	python script/eval_ultra.py --data_dir datasets/lanl/raw/ --log_file $log --output_dir results/lanl/ --device $device;
done
for log in output_ft/Ultra/OpTCRich/*/log.txt; do
	python script/eval_ultra.py --data_dir datasets/lanl-rich/raw/ --log_file $log --output_dir results/lanl-rich/ --device $device;
done
for log in output_ft/Ultra/LANL/*/log.txt; do
	python script/eval_ultra.py --data_dir datasets/optc/raw/ --log_file $log --output_dir results/optc/ --device $device;
done
for log in output_ft/Ultra/LANLRich/*/log.txt; do
	python script/eval_ultra.py --data_dir datasets/optc-rich/raw/ --log_file $log --output_dir results/optc-rich/ --device $device;
done

# Training from scratch
for dataset in LANL LANLRich OpTC OpTCRich; do
	for seed in 0 1 2 3 4 5 6 7 8 9; do
		python script/run.py -c config/transductive/pretraining.yaml --seed $seed --dataset $dataset --gpus [0] --root $(pwd)/datasets/ --output_dir $(pwd)/output_pt/;
	done
done

# Evaluation with training on same dataset
for log in output_pt/Ultra/LANL/*/log.txt; do
	python script/eval_ultra.py --data_dir datasets/lanl/raw/ --log_file $log --output_dir results/lanl/ --device $device;
done
for log in output_pt/Ultra/LANLRich/*/log.txt; do
	python script/eval_ultra.py --data_dir datasets/lanl-rich/raw/ --log_file $log --output_dir results/lanl-rich/ --device $device;
done
for log in output_pt/Ultra/OpTC/*/log.txt; do
	python script/eval_ultra.py --data_dir datasets/optc/raw/ --log_file $log --output_dir results/optc/ --device $device;
done
for log in output_pt/Ultra/OpTCRich/*/log.txt; do
	python script/eval_ultra.py --data_dir datasets/optc-rich/raw/ --log_file $log --output_dir results/optc-rich/ --device $device;
done

# Evaluation with training on other dataset

for log in output_pt/Ultra/OpTC/*/log.txt; do
	python script/eval_ultra.py --data_dir datasets/lanl/raw/ --log_file $log --output_dir results/lanl/ --device $device;
done
for log in output_pt/Ultra/OpTCRich/*/log.txt; do
	python script/eval_ultra.py --data_dir datasets/lanl-rich/raw/ --log_file $log --output_dir results/lanl-rich/ --device $device;
done
for log in output_pt/Ultra/LANL/*/log.txt; do
	python script/eval_ultra.py --data_dir datasets/optc/raw/ --log_file $log --output_dir results/optc/ --device $device;
done
for log in output_pt/Ultra/LANLRich/*/log.txt; do
	python script/eval_ultra.py --data_dir datasets/optc-rich/raw/ --log_file $log --output_dir results/optc-rich/ --device $device;
done