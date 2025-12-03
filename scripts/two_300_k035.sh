#!/bin/bash

total=$((2 * 4 * 2))
current=0

for k in 3 5; do 
	for s in random rag ic jacc; do
		for m in mistral qwen; do
			((current++))
			echo "====================================================="
			echo "[$current/$total] k=$k, strategy=$s, model=$m"
			echo "====================================================="
			python3 main.py \
				-m benchmark \
				-s $s \
				--model $m \
				-b 300 \
				-k $k
			if [ $? -ne 0 ]; then
				echo "ERROR: Experiment failed!"
				echo "k=$k, strategy=$s, model=$m"
				exit 1
			fi
		done
	done
done

echo ""
echo "✅ All $total experiments completed successfully!"
