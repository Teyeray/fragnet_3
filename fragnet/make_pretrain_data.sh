export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python data_create/create_pretrain_datasets.py --raw_data_path exps/pt/pvkp_0331/prepare_data/book_cleaned.csv --save_path pretrain_data/0331_book_full


#/AI4S/Users/yangrm/projects/fragnet_3/fragnet/exps/pt/pvkp_0331/prepare_data/book_cleaned.csv