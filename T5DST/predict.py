# Copyright (c) Facebook, Inc. and its affiliates

import os
import random
import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from transformers import (AdamW, T5Tokenizer, BartTokenizer, BartForConditionalGeneration,
                          T5ForConditionalGeneration, WEIGHTS_NAME, CONFIG_NAME)
from data_loader import prepare_testdata
from config import get_args
from evaluate import evaluate_metrics
import json
from tqdm import tqdm
from copy import deepcopy
import numpy as np
from collections import Counter

# def consistency_cross_entropy(lm_logits1, lm_logits2, threshold=0.4):
#     logsoftmax = torch.nn.LogSoftmax(dim=1)
#     softmax = torch.nn.Softmax(dim=1)

#     lm_logits1 = lm_logits1.squeeze()
#     lm_logits2 = lm_logits2.squeeze()
#     # (batch, vocab_size)
#     # give threshold
#     prob2 = softmax(lm_logits2)
#     # the result tuple of two output tensors (max, max_indices)
#     # print(torch.max(prob2, dim=1))
#     prob2_max, prob2_index = torch.max(prob2, dim=1)
#     valid = []
#     for i in range(prob2_max.shape[0]):
#         if (prob2_index[i]==5839 and prob2_max[i]>0.9) or (prob2_index[i]!=5839 and prob2_max[i]>threshold):
#             valid.append(1)
#         else:
#             valid.append(0)

#     #sharpening
#     soft_targets = softmax(lm_logits2/0.5)

#     loss_temp = torch.sum(- soft_targets * logsoftmax(lm_logits1), 1)
#     for i in range(prob2_max.shape[0]):
#         if valid[i]==0:
#             loss_temp[i]=0

#     return torch.mean(loss_temp)


class DST_Seq2Seq(pl.LightningModule):

    def __init__(self, args, tokenizer, model):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.lr = args["lr"]

    def training_step(self, batch, batch_idx):
        self.model.train()
        (loss), *_ = self.model(input_ids=batch["encoder_input"],
                                attention_mask=batch["attention_mask"],
                                lm_labels=batch["decoder_output"]
                                )

        # result = pl.TrainResult(loss)
        # result.log('train_loss', loss, on_epoch=True)
        return {'loss': loss, 'log': {'train_loss': loss}}
        # return result

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        (loss), *_ = self.model(input_ids=batch["encoder_input"],
                                attention_mask=batch["attention_mask"],
                                lm_labels=batch["decoder_output"]
                                )

        return {'val_loss': loss, 'log': {'val_loss': loss}}
        # return result

    def validation_epoch_end(self, outputs):
        val_loss_mean = sum([o['val_loss'] for o in outputs]) / len(outputs)
        # show val_loss in progress bar but only log val_loss
        results = {'progress_bar': {'val_loss': val_loss_mean.item()}, 'log': {'val_loss': val_loss_mean.item()},
                   'val_loss': val_loss_mean.item()}
        return results

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr, correct_bias=True)


def predict(args, tokenizer, model, test_loader, save_path, ALL_SLOTS, prefix="zeroshot"):
    save_path = os.path.join(save_path, "results")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    predictions = {}
    # to gpu
    # gpu = args["GPU"][0]
    device = torch.device("cuda:0")
    model.to(device)
    model.eval()

    with torch.no_grad():
        count = 0
        for batch in tqdm(test_loader):
            count += 1
            if (count == 5):
                break

            dst_outputs = model.generate(input_ids=batch["encoder_input"].to(device),
                                         attention_mask=batch["attention_mask"].to(
                device),
                eos_token_id=tokenizer.eos_token_id,
                max_length=200,
            )

            value_batch = tokenizer.batch_decode(
                dst_outputs, skip_special_tokens=True)

            for idx, value in enumerate(value_batch):
                dial_id = batch["ID"][idx]
                if dial_id not in predictions:
                    predictions[dial_id] = {}
                    predictions[dial_id]["domain"] = batch["domains"][idx][0]
                    predictions[dial_id]["turns"] = {}
                if batch["turn_id"][idx] not in predictions[dial_id]["turns"]:
                    predictions[dial_id]["turns"][batch["turn_id"][idx]] = {
                        "turn_belief": batch["turn_belief"][idx], "pred_belief": []}

                if value != "none":
                    predictions[dial_id]["turns"][batch["turn_id"][idx]]["pred_belief"].append(
                        str(batch["slot_text"][idx])+'#####'+str(value))

    with open(os.path.join(save_path, f"prediction.json"), 'w') as f:
        json.dump(predictions, f, indent=4)

    return predictions


def test(args, *more):
    args = vars(args)
    args["model_name"] = args["model_checkpoint"]+args["model_name"]+"_except_domain_"+args["except_domain"] + "_slotlang_" + \
        str(args["slot_lang"]) + "_lr_" + str(args["lr"]) + "_epoch_" + \
        str(args["n_epochs"]) + "_seed_" + str(args["seed"])
    # train!
    seed_everything(args["seed"])

    if "t5" in args["model_name"]:
        model = T5ForConditionalGeneration.from_pretrained(
            args["model_checkpoint"])
        tokenizer = T5Tokenizer.from_pretrained(
            args["model_checkpoint"], bos_token="[bos]", eos_token="[eos]", sep_token="[sep]")
        model.resize_token_embeddings(new_num_tokens=len(tokenizer))
    elif "bart" in args["model_name"]:
        model = BartForConditionalGeneration.from_pretrained(
            args["model_checkpoint"])
        tokenizer = BartTokenizer.from_pretrained(
            args["model_checkpoint"], bos_token="[bos]", eos_token="[eos]", sep_token="[sep]")
        model.resize_token_embeddings(new_num_tokens=len(tokenizer))

    task = DST_Seq2Seq(args, tokenizer, model)

    test_loader, ALL_SLOTS = prepare_testdata(
        args, tokenizer)

    #save model path
    save_path = os.path.join(args["saving_dir"], "t5-small")
    task.model = T5ForConditionalGeneration.from_pretrained(save_path)
    task.tokenizer = T5Tokenizer.from_pretrained(save_path)

    print("test start...")
    #evaluate model
    _ = predict(args, task.tokenizer, task.model,
                test_loader, save_path, ALL_SLOTS)


if __name__ == "__main__":
    args = get_args()
    if args.mode == "test":
        test(args)
