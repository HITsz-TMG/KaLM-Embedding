
# You can specify the Hugging Face Datasets cache path to better manage data, especially when you need to run code in a non-networked environment.
export HF_DATASETS_CACHE=


cd $(dirname "$0")/..


model_name_or_path=
output_dir=
eval_lang=en
master_port=12332

python evaluate/run_mteb.py --model_name_or_path ${model_name_or_path} --eval_lang ${eval_lang} --output_dir ${output_dir} --master_port ${master_port} --use_instruct 

