create_data:
	bash create_data.sh data-0625 T5DST/
train:
	bash train.sh
test:
	bash test.sh data/test_unseen_dials.json
	bash test.sh data/test_seen_dials.json
