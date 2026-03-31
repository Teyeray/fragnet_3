python data_create/create_finetune_datasets.py \
    --dataset_name gen \
    --train_path finetune_data/perovskite/train.csv \
    --val_path   finetune_data/perovskite/val.csv \
    --test_path  finetune_data/perovskite/test.csv \
    --output_dir finetune_data/perovskite_pkl \
    --target_name y \
    --frag_type brics