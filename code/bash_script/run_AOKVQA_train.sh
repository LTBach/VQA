data=4
# Train Knowledge Space
python main.py --gpu_id 1 --exp_name knowledge_space_aokvqa --exp_id W2V --fusion_model SAN --data_choice "${data}" --method_choice W2V --save_model 1

# Train Semantic Space
python main.py --gpu_id 1 --exp_name semantic_space_aokvqa --exp_id W2V --fusion_model SAN --data_choice "${data}" --method_choice W2V --save_model 1 --relation_map 1

# Train Object Space
python main.py --gpu_id 1 --exp_name object_space_aokvqa --exp_id W2V --fusion_model SAN --data_choice "${data}" --method_choice W2V --save_model 1 --fact_map 1
