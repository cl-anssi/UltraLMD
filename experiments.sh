lanl=/path/to/lanl/dataset/
optc=/path/to/optc/dataset/ecar/
python src/extract_optc.py $optc optc_redteam.csv --output_dir datasets/optc/
python src/extract_lanl.py $lanl --output_dir datasets/lanl/

device=cuda
# Experiments on OpTC
# UltraLMD++
python src/run_experiment.py datasets/optc/ results/optc/ --ckpt $(pwd)/ckpts/ultra_50g.pth --train_cutoff 1490 --num_context_graphs 100 --refine_scores --device $device
# UltraLMD + retrieval
python src/run_experiment.py datasets/optc/ results/optc/ --ckpt $(pwd)/ckpts/ultra_50g.pth --train_cutoff 1490 --num_context_graphs 100 --device $device
# UltraLMD + refinement
python src/run_experiment.py datasets/optc/ results/optc/ --ckpt $(pwd)/ckpts/ultra_50g.pth --train_cutoff 1490 --num_context_graphs -1 --refine_scores --device $device
# UltraLMD
python src/run_experiment.py datasets/optc/ results/optc/ --ckpt $(pwd)/ckpts/ultra_50g.pth --train_cutoff 1490 --num_context_graphs -1 --device $device

# Experiments on LANL
# UltraLMD++
python src/run_experiment.py datasets/lanl/ results/lanl/ --ckpt $(pwd)/ckpts/ultra_50g.pth --train_cutoff 41 --num_context_graphs 10 --refine_scores --device $device
# UltraLMD + retrieval
python src/run_experiment.py datasets/lanl/ results/lanl/ --ckpt $(pwd)/ckpts/ultra_50g.pth --train_cutoff 41 --num_context_graphs 10 --device $device
# UltraLMD + refinement
python src/run_experiment.py datasets/lanl/ results/lanl/ --ckpt $(pwd)/ckpts/ultra_50g.pth --train_cutoff 41 --num_context_graphs -1 --refine_scores --device $device
# UltraLMD
python src/run_experiment.py datasets/lanl/ results/lanl/ --ckpt $(pwd)/ckpts/ultra_50g.pth --train_cutoff 41 --num_context_graphs -1 --device $device
