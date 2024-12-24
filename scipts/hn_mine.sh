
# pip install faiss-gpu
# pip install FlagEmbedding==1.1

cd $(dirname "$0")/..



# Process sinlge file

model_name_or_path=
pooling_method=cls
input_file=
output_file=

python3 -m train.hn_mine \
    --model_name_or_path ${model_name_or_path} \
    --pooling_method ${pooling_method} \
    --input_file ${input_file} \
    --output_file ${output_file} \
    --negative_number 7 \
    --range_for_sampling 50-100 \
    --filter_topk 50



# Process the directory

model_name_or_path=
pooling_method=cls
input_dir=
output_dir=

for input_file in "$input_dir"/*; do
    filename=$(basename "$input_file")
    output_file="$output_dir/$filename"

    if [ -f "$output_file" ]; then
        echo "Output file $output_file already exists. Skipping..."
        continue
    fi

    echo "Processing ${input_file}"
    
     python3 -m train.hn_mine \
        --model_name_or_path ${model_name_or_path} \
        --pooling_method ${pooling_method} \
        --input_file ${input_file} \
        --output_file ${output_file} \
        --negative_number 7 \
        --range_for_sampling 50-100 \
        --filter_topk 50
done
