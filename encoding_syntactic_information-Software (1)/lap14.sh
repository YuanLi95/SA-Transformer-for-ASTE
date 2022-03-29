#! /bin/bash

for nhop in   1 2 3 
do 
	for alpha_adjacent in 0.1 0.3 0.5
		do
			for weight_edge in 0.3 0.6 1.0
			do
				for num_syntransformer_layers in 1 2 3
				do	
					for position_embed_dim in 100 50
					do
						for dependency_embed_dim in 100 200
						do
						for seed in 19 97 199
						do


					python main.py --batch_size=30 --nhop $nhop --seed $seed --dataset "lap14"  --alpha_adjacent $alpha_adjacent --nhop $nhop --weight_edge $weight_edge\
							--num_syntransformer_layers $num_syntransformer_layers --position_embed_dim $position_embed_dim --position_embed_dim $position_embed_dim
						done
						done
					done
				done
			done
		done
done 
