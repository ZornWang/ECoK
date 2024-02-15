import csv
import argparse
import os
from ast import literal_eval
from collections import defaultdict
from transformers import AutoTokenizer
import pickle
import pandas as pd


def read_ERC_datas(root, dataset_name, split):
    with open(os.path.join(root, dataset_name, split + ".pkl"), "rb") as f:
        raw_data = pickle.load(f)
    return raw_data


def read_RECCON_datas(root, dataset_name, split):
    with open(os.path.join(root, dataset_name, split + ".pkl"), "rb") as f:
        target_context, speakers, cause_labels, emotions, ids, _ = pickle.load(f)
    return target_context, speakers, cause_labels, emotions, ids


def get_ngrams(utter, n):
    # utter: a list of tokens
    # n: up to n-grams
    total = []
    for i in range(len(utter)):
        for j in range(i, max(i - n, -1), -1):
            total.append("_".join(utter[j : i + 1]).lower())
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="RECCON")
    parser.add_argument("--root", default="comet_origin_enhanced_data")
    parser.add_argument("--split", default="train")
    parser.add_argument("--n", default=4)
    parser.add_argument("--tokenizer_dir", default="bert-base-uncased")
    args = parser.parse_args()

    dataset = args.dataset
    root = args.root
    n = args.n
    split = args.split

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    conceptnet = pd.read_csv("conceptnet_en.csv")
    item1 = conceptnet["item1"]
    relation = conceptnet["relation"]
    item2 = conceptnet["item2"]
    weights = conceptnet["weight"]

    total_num = 0
    utt_num = 0

    print("Loading dataset...")
    if dataset == "RECCON":
        target_context, speakers, cause_labels, emotions, ids = read_RECCON_datas(
            root, dataset, split
        )
        new_data = {}
        k = 0
        for id in ids:
            dialogue = target_context[id]
            dialogue_know = []
            for sen in dialogue:
                sentence = sen.lower()
                # split_sentence = sentence.split()
                knowledge = []
                for i1, r, i2, w in zip(item1, relation, item2, weights):
                    i1 = str(i1)
                    if "_" in i1:
                        i1 = " ".join(i1.split("_"))
                    if i1 in sentence:
                        if w < 2.01:
                            continue
                        if "_" in i2:
                            i2 = " ".join(i2.split("_"))
                        knowledge.append([i1 + "," + r + "," + i2, w])
                total_num += len(knowledge)
                utt_num += 1
                if utt_num % 10 == 0:
                    print(utt_num)
                knowledge = sorted(knowledge, key=lambda x: (x[1]), reverse=True)
                dialogue_know.append(knowledge)
                k += 1
            new_data[id] = dialogue_know
        print(new_data)
        pickle.dump(
            [target_context, speakers, cause_labels, emotions, ids, new_data],
            open("{}_{}".format(dataset, split) + ".pkl", "wb+"),
        )
    else:
        new_data = []
        data_raw = read_ERC_datas(root, dataset, split)
        for dialogue in data_raw:
            new_dialogue = []
            for sen in dialogue:
                new_sen = {}
                new_sen["text"] = sen["text"]
                new_sen["label"] = sen["label"]
                new_sen["speaker"] = sen["speaker"]

                sentence = sen["text"].lower()
                split_sentence = sentence.split()
                knowledge = []
                for i1, r, i2, w in zip(item1, relation, item2, weights):
                    i1 = str(i1)
                    if "_" in i1:
                        i1 = " ".join(i1.split("_"))
                        if i1 in sentence:
                            if w < 2.1:
                                continue
                            if "_" in i2:
                                i2 = " ".join(i2.split("_"))
                            knowledge.append([i1 + "," + r + "," + i2, w])
                    else:
                        if i1 in split_sentence:
                            if w < 2.1:
                                continue
                            if "_" in i2:
                                i2 = " ".join(i2.split("_"))
                            knowledge.append([i1 + "," + r + "," + i2, w])
                total_num += len(knowledge)
                utt_num += 1
                if utt_num % 10 == 0:
                    print(utt_num)
                knowledge = sorted(knowledge, key=lambda x: (x[1]), reverse=True)
                new_sen["knowledge"] = knowledge
                new_dialogue.append(new_sen)
            new_data.append(new_dialogue)
            pickle.dump(new_data, open("{}_{}.pkl".format(dataset, split), "wb+"))
    print("The average knowledge num: {}".format(total_num / utt_num))

    """print("Loading conceptnet...")
    csv_reader = csv.reader(open("./conceptnet-assertions-5.7.0.csv", "r"), delimiter="\t")
    # dicts = defaultdict(set)
    output = {'item1': [], 'relation': [], 'item2': [], 'weight':[]}
    for i, row in enumerate(csv_reader):
        if i % 1000000 == 0:
            print("Processed {0} rows".format(i))

        lang1 = row[2].split("/")[2]
        lang2 = row[3].split("/")[2]
        if lang1 == lang2 == 'en':
            item1 = row[2].split("/")[3].strip().lower()
            item2 = row[3].split("/")[3].strip().lower()
            if len(item1) <= 1 or item1 == item2:
                continue
            relation = row[1].split("/")[2]
            weight = float(row[4].split(',')[-1].split(':')[-1].strip()[:-1])
            output['item1'].append(item1)
            output['item2'].append(item2)
            output['relation'].append(relation)
            output['weight'].append(weight)
            print("{}/{}/{}/{}".format(item1, relation, item2, weight))
    results = pd.DataFrame(output, index=None)
    results.to_csv('conceptnet_en.csv', index=False)"""
