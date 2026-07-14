先生成da3:

切到conda activate da3

python scripts/run_da3.py \
  --image_dir ./data/example/images \
  --output_dir ./da3_outputs/example \
  --model_path /home/mengxiangting/nas/mengxt/Projects/Depth-Anything-3/DA3_GAINT


然后：

conda deactivate
conda activate sam3d-objects

python run_inference_weighted.py \
  --input_path ./data/example \
  --mask_prompt stuffed_toy \
  --da3_output ./da3_outputs/example/da3_output.npz