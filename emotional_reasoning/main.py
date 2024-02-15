"""Full training script"""
import os
import logging
import random
import numpy as np
import torch
from torch import nn
from transformers import get_linear_schedule_with_warmup, AdamW
import json
from model import VIBERC, CasualVIBERC, RobertaClassifier, CasualRobertaClassifier
from utils import (
    ErcTextDataset,
    RECCONTextDataset,
    get_num_classes,
    replace_for_robust_eval,
)
import math
import argparse
import yaml
import pickle
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
import torch.cuda.amp.grad_scaler as grad_scaler
import torch.cuda.amp.autocast_mode as autocast_mode


def train_or_eval(
    epoch,
    model,
    optimizer,
    scheduler,
    loss_function,
    mode,
    data,
    batch_size,
    cuda,
    alpha,
    beta,
    scaler,
    kl_weights_dict,
):

    random.shuffle(data)
    if mode == "train":
        model.train()
    else:
        model.eval()
    predicts = []
    ground_truth = []
    loss_list = []
    latent_param_dict = {}
    for item in kl_weights_dict.keys():
        latent_param_dict[item] = []
    label_records = []
    f_cos_simi = []
    b_cos_simi = []
    for i in range(0, len(data), batch_size):
        if mode == "train":
            optimizer.zero_grad()
        bs_data = data[i : min(i + batch_size, len(data))]
        input_data = pad_sequence(
            [torch.LongTensor(item["input_ids"]) for item in bs_data],
            batch_first=True,
            padding_value=1,
        )
        masks = pad_sequence(
            [torch.LongTensor(item["attention_mask"]) for item in bs_data],
            batch_first=True,
            padding_value=0,
        )

        """input_data = []
        masks = []
        for item in bs_data:
            input_data += item['input_ids']
            masks += item['attention_mask']
        input_data = pad_sequence([torch.LongTensor(item) for item in input_data], batch_first=True,
                                  padding_value=1)
        masks = pad_sequence([torch.LongTensor(item) for item in masks], batch_first=True,
                             padding_value=0)"""

        utt_pos_masks = []
        for i in range(len(bs_data[0]["pos_masks"])):
            utt_pos_masks.append(
                pad_sequence(
                    [torch.LongTensor(item["pos_masks"][i]) for item in bs_data],
                    batch_first=True,
                    padding_value=0,
                )
            )
        utt_pos_masks = torch.stack(utt_pos_masks, dim=0)

        comet_features = torch.stack(
            [item["comet_features"]["feature"] for item in bs_data], dim=0
        )
        comet_masks = torch.stack(
            [item["comet_features"]["mask"] for item in bs_data], dim=0
        )
        labels = torch.LongTensor([item["label"] for item in bs_data])
        # comet_sent_labels = torch.LongTensor([item['comet_features']['comet_sent_labels'] for item in bs_data])
        if cuda:
            input_data = input_data.cuda()
            masks = masks.cuda()
            utt_pos_masks = utt_pos_masks.cuda()
            labels = labels.cuda()

            comet_features = comet_features.cuda()
            comet_masks = comet_masks.cuda()
            # comet_sent_labels = comet_sent_labels.cuda()

        outputs, latent_params, f_cos, b_cos = model(
            input_data, masks, utt_pos_masks, comet_features, comet_masks
        )
        f_cos_simi.append(f_cos)
        b_cos_simi.append(b_cos)
        # outputs = model(input_data, masks, utt_pos_masks)
        ce_loss = loss_function(outputs, labels)
        # utt_comet_ce_loss = loss_function(comet_utt_output.view(-1, comet_utt_output.shape[-1]), comet_sent_labels.view(-1))
        """kl_loss = losses.compute_kl_divergence_losses(
            model, latent_params, kl_weights_dict)['total_weighted_kl']"""
        # loss = ce_loss + kl_loss
        loss = ce_loss

        if mode == "train":
            loss.backward()
            optimizer.step()
            scheduler.step()

        ground_truth += labels.cpu().numpy().tolist()
        # ground_truth += o_labels
        predicts += torch.argmax(outputs, dim=1).cpu().numpy().tolist()
        # predicts += compute_predicts(outputs.cpu(), label_VAD)
        loss_list.append(loss.item())
        if mode == "eval":
            for item in latent_params.keys():
                k = torch.cat(
                    [
                        latent_params[item].mu.cpu().detach(),
                        torch.exp(latent_params[item].logvar.cpu().detach()),
                        latent_params[item].z.cpu().detach(),
                    ],
                    dim=-1,
                )
                latent_param_dict[item].append(k)
            label_records += labels.cpu().tolist()

    avg_loss = round(np.sum(loss_list) / len(loss_list), 4)
    avg_accuracy = round(accuracy_score(ground_truth, predicts) * 100, 2)
    weighted_f1 = round(f1_score(ground_truth, predicts, average="weighted") * 100, 2)
    if args["DATASET"] == "DailyDialog":
        micro_f1 = round(
            f1_score(ground_truth, predicts, average="micro", labels=list(range(1, 7)))
            * 100,
            2,
        )
    else:
        micro_f1 = round(f1_score(ground_truth, predicts, average="micro") * 100, 2)
    # micro_f1 = round(f1_score(ground_truth, predicts, average='micro', labels=list(range(1, 7))) * 100, 2)
    macro_f1 = round(f1_score(ground_truth, predicts, average="macro") * 100, 2)

    result_name = "result.xlsx"
    if os.path.exists(result_name):
        result = pd.read_excel(result_name)
    else:
        result = pd.DataFrame(
            columns=[
                "epochs",
                "mode",
                "avg_loss",
                "weighted_f1",
                "micro_f1",
                "macro_f1",
                "dataset",
                "experiment",
                "params",
            ]
        )
    params = ",".join([key + ":" + str(value) for key, value in args.items()])
    new_df = pd.DataFrame(
        [
            [
                epoch,
                mode,
                avg_loss,
                weighted_f1,
                micro_f1,
                macro_f1,
                args["DATASET"],
                args["experiment"],
                params,
            ]
        ],
        columns=[
            "epochs",
            "mode",
            "avg_loss",
            "weighted_f1",
            "micro_f1",
            "macro_f1",
            "dataset",
            "experiment",
            "params",
        ],
    )
    result = pd.concat([result, new_df], axis=0)
    result.to_excel(result_name, index=False)

    if mode == "train":
        print(
            "For epoch {}, train loss:{}, weighted F1 {}, micro F1 {}, macro F1 {}".format(
                epoch, avg_loss, weighted_f1, micro_f1, macro_f1
            )
        )
    if mode == "dev":
        print(
            "For epoch {}, dev loss:{}, weighted F1 {}, micro F1 {}, macro F1 {}".format(
                epoch, avg_loss, weighted_f1, micro_f1, macro_f1
            )
        )
    if mode == "test" or mode == "eval":
        print(
            "For epoch {}, test loss:{}, weighted F1 {}, micro F1 {}, macro F1 {}".format(
                epoch, avg_loss, weighted_f1, micro_f1, macro_f1
            )
        )
        f_cos_simi = torch.cat(f_cos_simi, dim=0).transpose(1, 0).tolist()
        b_cos_simi = torch.cat(b_cos_simi, dim=0).transpose(1, 0).tolist()
        rep_similarity_path = os.path.join("./rep_similarity", args["DATASET"])
        if not os.path.exists(rep_similarity_path):
            os.makedirs(rep_similarity_path)
        metric = {
            "weighted_f1": weighted_f1,
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
        }
        pickle.dump(
            metric, open("{}/metric_{}.pkl".format(rep_similarity_path, epoch), "wb+")
        )
        pickle.dump(
            f_cos_simi, open("{}/f_{}.pkl".format(rep_similarity_path, epoch), "wb+")
        )
        pickle.dump(
            b_cos_simi, open("{}/b_{}.pkl".format(rep_similarity_path, epoch), "wb+")
        )
        if args["DATASET"] == "DailyDialog":
            print(
                f1_score(ground_truth, predicts, average=None, labels=list(range(1, 7)))
            )
        else:
            print(f1_score(ground_truth, predicts, average=None))


def casual_train_or_eval(
    epoch,
    model,
    optimizer,
    scheduler,
    loss_function,
    mode,
    data,
    batch_size,
    cuda,
    alpha,
    beta,
    scaler,
    kl_weights_dict,
):
    random.shuffle(data)
    if mode == "train":
        model.train()
    else:
        model.eval()
    predicts = []
    ground_truth = []
    loss_list = []
    latent_param_dict = {}
    bceloss = nn.BCELoss()
    for item in kl_weights_dict.keys():
        latent_param_dict[item] = []
    label_records = []
    for i in range(0, len(data), batch_size):
        if mode == "train":
            optimizer.zero_grad()
        bs_data = data[i]
        # input_data = pad_sequence([torch.LongTensor(item['input_ids']) for item in bs_data], batch_first=True, padding_value=1)
        # masks = pad_sequence([torch.LongTensor(item['attention_mask']) for item in bs_data], batch_first=True, padding_value=0)
        input_data = torch.LongTensor(bs_data["input_ids"])
        masks = torch.LongTensor(bs_data["attention_mask"])

        """input_data = []
        masks = []
        for item in bs_data:
            input_data += item['input_ids']
            masks += item['attention_mask']
        input_data = pad_sequence([torch.LongTensor(item) for item in input_data], batch_first=True,
                                  padding_value=1)
        masks = pad_sequence([torch.LongTensor(item) for item in masks], batch_first=True,
                             padding_value=0)"""

        pos_masks = bs_data["pos_masks"]
        comet_features = torch.stack(
            [torch.FloatTensor(item) for item in bs_data["comet_features"]], dim=0
        )
        labels = torch.FloatTensor(bs_data["label"])
        emolabels = torch.LongTensor(bs_data["emolabel"])
        if cuda:
            input_data = input_data.cuda()
            masks = masks.cuda()
            pos_masks = pos_masks.cuda()
            labels = labels.cuda()

            comet_features = comet_features.cuda()
            emolabels = emolabels.cuda()

        emo_outputs, cause_outputs, latent_params = model(
            input_data, masks, pos_masks, comet_features
        )
        # emo_outputs, cause_outputs = model(input_data, masks, pos_masks)
        emo_loss = loss_function(emo_outputs, emolabels)
        cause_loss = bceloss(cause_outputs, labels)
        # utt_comet_ce_loss = loss_function(comet_utt_output.view(-1, comet_utt_output.shape[-1]), comet_sent_labels.view(-1))
        """kl_loss = losses.compute_kl_divergence_losses(
            model, latent_params, kl_weights_dict)['total_weighted_kl']"""
        loss = cause_loss + emo_loss * alpha
        # loss = cause_loss + alpha*emo_loss

        if mode == "train":
            loss.backward()
            optimizer.step()
            scheduler.step()

        ground_truth += labels.cpu().numpy().tolist()
        # ground_truth += o_labels
        predicts += torch.gt(cause_outputs, 0.5).long().cpu().numpy().tolist()
        # predicts += compute_predicts(outputs.cpu(), label_VAD)
        loss_list.append(loss.item())
        if mode == "eval":
            for item in latent_params.keys():
                k = torch.cat(
                    [
                        latent_params[item].mu.cpu().detach(),
                        torch.exp(latent_params[item].logvar.cpu().detach()),
                        latent_params[item].z.cpu().detach(),
                    ],
                    dim=-1,
                )
                latent_param_dict[item].append(k)
            label_records += labels.cpu().tolist()

    avg_loss = round(np.sum(loss_list) / len(loss_list), 4)
    avg_accuracy = round(accuracy_score(ground_truth, predicts) * 100, 2)
    weighted_f1 = round(f1_score(ground_truth, predicts, average="weighted") * 100, 2)
    micro_f1 = round(f1_score(ground_truth, predicts, average="micro") * 100, 2)
    # micro_f1 = round(f1_score(ground_truth, predicts, average='micro', labels=list(range(1, 7))) * 100, 2)
    macro_f1 = round(f1_score(ground_truth, predicts, average="macro") * 100, 2)
    if mode == "train":
        print(
            "For epoch {}, train loss:{}, weighted F1 {}, micro F1 {}, macro F1 {}".format(
                epoch, avg_loss, weighted_f1, micro_f1, macro_f1
            )
        )
    if mode == "dev":
        print(
            "For epoch {}, dev loss:{}, weighted F1 {}, micro F1 {}, macro F1 {}".format(
                epoch, avg_loss, weighted_f1, micro_f1, macro_f1
            )
        )
    if mode == "test" or mode == "eval":
        print(
            "For epoch {}, test loss:{}, weighted F1 {}, micro F1 {}, macro F1 {}".format(
                epoch, avg_loss, weighted_f1, micro_f1, macro_f1
            )
        )
        print(f1_score(ground_truth, predicts, average=None))


def main(
    CUDA: bool,
    LR: float,
    SEED: int,
    DATASET: str,
    BATCH_SIZE: int,
    model_checkpoint: str,
    speaker_mode: str,
    num_past_utterances: int,
    num_future_utterances: int,
    NUM_TRAIN_EPOCHS: int,
    WEIGHT_DECAY: float,
    WARMUP_RATIO: float,
    COMET_WIN_SIZE: int,
    **kwargs,
):

    # CUDA = False
    ROOT_DIR = args["ROOT_DIR"]
    NUM_CLASS = get_num_classes(DATASET)
    lr = float(LR)
    # label_VAD = get_label_VAD(DATASET)

    if DATASET == "RECCON":
        ds_train = RECCONTextDataset(
            DATASET=DATASET,
            SPLIT="train",
            speaker_mode=speaker_mode,
            model_checkpoint=model_checkpoint,
            ROOT_DIR=ROOT_DIR,
            SEED=SEED,
        )
        ds_val = RECCONTextDataset(
            DATASET=DATASET,
            SPLIT="val",
            speaker_mode=speaker_mode,
            model_checkpoint=model_checkpoint,
            ROOT_DIR=ROOT_DIR,
            SEED=SEED,
        )
        ds_test = RECCONTextDataset(
            DATASET=DATASET,
            SPLIT="test",
            speaker_mode=speaker_mode,
            model_checkpoint=model_checkpoint,
            ROOT_DIR=ROOT_DIR,
            SEED=SEED,
        )
        model = CasualVIBERC(args, NUM_CLASS)
        # model = CasualRobertaClassifier(args, NUM_CLASS)

    else:
        ds_train = ErcTextDataset(
            DATASET=DATASET,
            SPLIT="train",
            speaker_mode=speaker_mode,
            num_past_utterances=num_past_utterances,
            num_future_utterances=num_future_utterances,
            model_checkpoint=model_checkpoint,
            ROOT_DIR=ROOT_DIR,
            SEED=SEED,
            COMET_WIN_SIZE=COMET_WIN_SIZE,
        )
        ds_val = ErcTextDataset(
            DATASET=DATASET,
            SPLIT="val",
            speaker_mode=speaker_mode,
            num_past_utterances=num_past_utterances,
            num_future_utterances=num_future_utterances,
            model_checkpoint=model_checkpoint,
            ROOT_DIR=ROOT_DIR,
            SEED=SEED,
            COMET_WIN_SIZE=COMET_WIN_SIZE,
        )
        ds_test = ErcTextDataset(
            DATASET=DATASET,
            SPLIT="test",
            speaker_mode=speaker_mode,
            num_past_utterances=num_past_utterances,
            num_future_utterances=num_future_utterances,
            model_checkpoint=model_checkpoint,
            ROOT_DIR=ROOT_DIR,
            SEED=SEED,
            COMET_WIN_SIZE=COMET_WIN_SIZE,
        )
        model = VIBERC(args, NUM_CLASS)
        # model = RobertaClassifier(args, NUM_CLASS)

    tr_data = ds_train.inputs_
    dev_data = ds_val.inputs_
    test_data = ds_test.inputs_
    if args["robust_rate"] != 0.0:
        tr_data = replace_for_robust_eval(tr_data, args["robust_rate"], NUM_CLASS)

    if args["mode"] != "train":
        model.load_state_dict(torch.load(args["model_load_path"]))
        model.eval()

    # kl_weights_dict = {'utt': args['utt_kl_weight'], 'comet_utt': args['comet_utt_kl_weight']}
    kl_weights_dict = {"utt": args["utt_kl_weight"]}

    loss_function = nn.CrossEntropyLoss(ignore_index=-1)
    if CUDA:
        model.cuda()

    if args["mode"] == "train":
        """gnn_params = list(map(id, model.gnn.parameters()))
        base_params = filter(lambda p: id(p) not in gnn_params,
                             model.parameters())
        optimizer = AdamW([{'params': base_params}, {'params': model.gnn.parameters(), 'lr':lr*100}]
                          , lr=lr, weight_decay=WEIGHT_DECAY)"""
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

        """Use linear scheduler."""
        s_total_steps = float(NUM_TRAIN_EPOCHS * len(ds_train.inputs_)) / BATCH_SIZE
        scheduler = get_linear_schedule_with_warmup(
            optimizer, int(s_total_steps * WARMUP_RATIO), math.ceil(s_total_steps)
        )
        # total_steps = math.ceil(float(args['NUM_TRAIN_EPOCHS'] * len(ds_train.inputs_)) / BATCH_SIZE)

        """Due to the limitation of computational resources, we use mixed floating point precision."""
        scaler = grad_scaler.GradScaler()
        # loss_function = nn.MSELoss()
        # loss_function = EMDLoss(args, label_type='single', label_VAD=label_VAD)

        for n in range(NUM_TRAIN_EPOCHS):
            # steps = n * math.ceil(float(len(tr_data)) / BATCH_SIZE)
            if DATASET == "RECCON":
                casual_train_or_eval(
                    n,
                    model,
                    optimizer,
                    scheduler,
                    loss_function,
                    "train",
                    tr_data,
                    BATCH_SIZE,
                    CUDA,
                    args["alpha"],
                    args["beta"],
                    scaler,
                    kl_weights_dict,
                )
                casual_train_or_eval(
                    n,
                    model,
                    optimizer,
                    scheduler,
                    loss_function,
                    "dev",
                    dev_data,
                    BATCH_SIZE,
                    CUDA,
                    args["alpha"],
                    args["beta"],
                    scaler,
                    kl_weights_dict,
                )
                casual_train_or_eval(
                    n,
                    model,
                    optimizer,
                    scheduler,
                    loss_function,
                    "test",
                    test_data,
                    BATCH_SIZE,
                    CUDA,
                    args["alpha"],
                    args["beta"],
                    scaler,
                    kl_weights_dict,
                )
                torch.save(
                    model.state_dict(),
                    args["model_save_dir"] + "/model_state_dict_" + str(n) + ".pth",
                )
                print("-------------------------------")
            else:
                train_or_eval(
                    n,
                    model,
                    optimizer,
                    scheduler,
                    loss_function,
                    "train",
                    tr_data,
                    BATCH_SIZE,
                    CUDA,
                    args["alpha"],
                    args["beta"],
                    scaler,
                    kl_weights_dict,
                )
                train_or_eval(
                    n,
                    model,
                    optimizer,
                    scheduler,
                    loss_function,
                    "dev",
                    dev_data,
                    BATCH_SIZE,
                    CUDA,
                    args["alpha"],
                    args["beta"],
                    scaler,
                    kl_weights_dict,
                )
                train_or_eval(
                    n,
                    model,
                    optimizer,
                    scheduler,
                    loss_function,
                    "test",
                    test_data,
                    BATCH_SIZE,
                    CUDA,
                    args["alpha"],
                    args["beta"],
                    scaler,
                    kl_weights_dict,
                )
                torch.save(
                    model.state_dict(),
                    args["model_save_dir"] + "/model_state_dict_" + str(n) + ".pth",
                )
                print("-------------------------------")
    else:
        if DATASET == "RECCON":
            casual_train_or_eval(
                None,
                model,
                None,
                None,
                loss_function,
                "eval",
                test_data,
                BATCH_SIZE,
                CUDA,
                args["alpha"],
                args["beta"],
                None,
                kl_weights_dict,
            )
        else:
            train_or_eval(
                None,
                model,
                None,
                None,
                loss_function,
                "eval",
                test_data,
                BATCH_SIZE,
                CUDA,
                args["alpha"],
                args["beta"],
                None,
                kl_weights_dict,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="erc RoBERTa text huggingface training"
    )
    parser.add_argument("--DATASET", type=str, default="IEMOCAP")
    parser.add_argument("--CUDA", action="store_true")
    parser.add_argument("--model_checkpoint", type=str, default="roberta-base")
    parser.add_argument("--speaker_mode", type=str, default="title")
    parser.add_argument("--num_past_utterances", type=int, default=1000)
    parser.add_argument("--num_future_utterances", type=int, default=1000)
    parser.add_argument("--BATCH_SIZE", type=int, default=4)
    parser.add_argument("--LR", type=float, default=1e-5)
    parser.add_argument("--HP_ONLY_UPTO", type=int, default=10)
    parser.add_argument("--NUM_TRAIN_EPOCHS", type=int, default=10)
    parser.add_argument("--WEIGHT_DECAY", type=float, default=0.01)
    parser.add_argument("--WARMUP_RATIO", type=float, default=0.2)
    parser.add_argument("--HP_N_TRIALS", type=int, default=5)
    parser.add_argument("--OUTPUT-DIR", type=str, default="./output")
    parser.add_argument("--SEED", type=int, default=42)
    parser.add_argument(
        "--alpha", default=0.8, type=float, help="The loss coefficient."
    )
    parser.add_argument(
        "--beta", default=0.5, type=float, help="The con loss coefficient."
    )
    parser.add_argument(
        "--utt_kl_weight",
        default=0.001,
        type=float,
        help="The kl loss coefficient for utterance representations.",
    )
    parser.add_argument(
        "--comet_utt_kl_weight",
        default=0.005,
        type=float,
        help="The kl loss coefficient for utterance representations.",
    )
    parser.add_argument(
        "--mi_loss_weight", default=0.001, type=float, help="The mi loss coefficient."
    )
    parser.add_argument(
        "--mode", default="train", type=str, help="Training or eval mode."
    )
    parser.add_argument(
        "--model_save_dir",
        default="./model_save_dir",
        type=str,
        help="Save the trained model in this dir.",
    )
    parser.add_argument(
        "--model_load_path", type=str, help="Load the trained model from here."
    )
    parser.add_argument(
        "--latent_param_save_path",
        default="./latent_save_dir/",
        type=str,
        help="Save the latent params here.",
    )
    parser.add_argument(
        "--mi_loss", action="store_true", help="Whether add the mi loss."
    )
    parser.add_argument(
        "--robust_rate",
        default=0.0,
        type=float,
        help="Replace rate for robustness evaluation",
    )
    parser.add_argument(
        "--COMET_WIN_SIZE", default=5, type=int, help="Window size of COMET knowledge."
    )
    parser.add_argument(
        "--COMET_HIDDEN_SIZE", default=768, type=int, help="COMET hidden size."
    )
    parser.add_argument(
        "--CONV_NAME", default="multidim_hgt", type=str, help="The HGT type."
    )
    parser.add_argument(
        "--ROOT_DIR", default="./comet_enhanced_data", type=str, help="The HGT type."
    )
    parser.add_argument("--experiment", default=1, type=int, help="experiment number.")

    args = parser.parse_args()
    args = vars(args)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args["CUDA"] is True else "cpu"
    )
    args["n_gpu"] = torch.cuda.device_count()
    args["device"] = device
    args["model_save_dir"] = os.path.join(args["model_save_dir"], args["DATASET"])
    if not os.path.exists(args["model_save_dir"]):
        os.makedirs(args["model_save_dir"])

    """with open('./train-erc-text.yaml', 'r') as stream:
        args_ = yaml.load(stream, Loader=yaml.FullLoader)

    for key, val in args_.items():
        args[key] = val"""

    logging.info(f"arguments given to {__file__}: {args}")
    main(**args)
