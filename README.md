# Leveraging Slot Descriptions for Zero-Shot Cross-Domain Dialogue State Tracking

## Abstract:
Zero-shot cross-domain dialogue state tracking (DST) enables us to handle task-oriented dialogue in unseen domains without the expense of collecting in-domain data. In this paper, we propose a slot description enhanced generative approach for zero-shot cross-domain DST. Specifically, our model first encodes dialogue context and slots with a pre-trained self-attentive encoder, and generates slot values in an auto-regressive manner. In addition, we incorporate Slot Type Informed Descriptions that capture the shared information across slots to facilitate cross-domain knowledge transfer. Experimental results on the MultiWOZ dataset show that our proposed method significantly improves existing state-of-the-art results in the zero-shot cross-domain setting.


## Dependency
Check the packages needed or simply run the command
```console
pip install -r requirements.txt
```

## Download model
```
bash download.sh
```

## Training and predicting
First put data-0625/ under current folder.
Note: data-0625/ should contain train/, dev/ test_seen/, test_unseen/, schema.json these folders and file.

### Data preprocessing
**Dataset**
This command will convert original data into T5DST's input form and put the converted data into T5DST folder.
```console
make create_data
```

### Training
**Full-shot training**
```console
make train
```
### Generate answer
For generating the final anser, you need to run both **Predicting** and **Postprocessing** part and the result will be **seen_pred.csv** and **unseen_pred.csv** at the current directory.

**Predicting**
```console
make test
```

**Postproecessing**
```console
make posprocess
```



# Citations
```
@inproceedings{lin2021leveraging,
  title={Leveraging Slot Descriptions for Zero-Shot Cross-Domain Dialogue StateTracking},
  author={Lin, Zhaojiang and Liu, Bing and Moon, Seungwhan and Crook, Paul A and Zhou, Zhenpeng and Wang, Zhiguang and Yu, Zhou and Madotto, Andrea and Cho, Eunjoon and Subba, Rajen},
  booktitle={Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  pages={5640--5648},
  year={2021}
}
```