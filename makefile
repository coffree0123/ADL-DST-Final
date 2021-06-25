create_data:
	bash create_data.sh data-0625 T5DST/
train:
	bash train.sh
test:
	bash test.sh data/train_dials.json
