for dataset in lanl lanl-rich optc optc-rich; do
	for seed in 0 1 2 3 4 5 6 7 8 9; do
		python script/baseline.py --seed $seed --data_dir datasets/$dataset/raw/ --output_dir results/$dataset/;
	done
done