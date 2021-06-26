cd T5DST
python predict.py --test_file $1 --output_file $2 --slot_lang human --GPU 1 --test_batch_size 64 --mode test