import pickle
import argparse
import torch
import os
from transformers import AutoTokenizer, AutoModel

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def triple_to_sen(triple, transfer_dict):
    triple = triple.split(",")
    return " ".join([triple[0].strip(), transfer_dict[triple[1]], triple[2].strip()])


def add_relations(dataset, split):
    # data = pickle.load(open('{}_{}.pkl'.format(dataset, split), 'rb'))
    target_context, speakers, cause_labels, emotions, ids, extracted_data = pickle.load(
        open("{}_{}.pkl".format(dataset, split), "rb")
    )
    relation_dict = pickle.load(open("relation_convert_dict.pkl", "rb"))
    print(relation_dict)
    relation_set = set([])

    for dialogue in extracted_data.values():
        for sen in dialogue:
            for item in sen:
                relation = item[0].split(",")[1]
                if relation not in relation_dict.keys():
                    relation_set.add(relation)

    print(len(relation_set))
    for item in relation_set:
        print(item)
        r = input("The convert is:")
        relation_dict[item] = r.strip()
    pickle.dump(relation_dict, open("relation_convert_dict.pkl", "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="IEMOCAP")
    parser.add_argument("--cuda", default=False)
    parser.add_argument("--split", default="train")
    parser.add_argument("--tokenizer_dir", default="roberta-base")
    args = parser.parse_args()

    dataset = args.dataset
    split = args.split
    cuda = args.cuda
    tokenizer_dir = args.tokenizer_dir

    encoder = AutoModel.from_pretrained(tokenizer_dir)
    if cuda:
        encoder.cuda()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    new_data = []
    relation_dict = pickle.load(open("relation_convert_dict.pkl", "rb"))
    # relation_dict = {}
    num = 0
    relation_set = set([])
    none_rep = tokenizer("", return_tensors="pt")["input_ids"]
    if cuda:
        none_rep = none_rep.cuda()
    embeddings = encoder(none_rep)
    none_rep = (
        embeddings.last_hidden_state.squeeze(0)[0, :].detach().cpu().numpy().tolist()
    )

    num = 0
    if dataset == "RECCON":
        (
            target_context,
            speakers,
            cause_labels,
            emotions,
            ids,
            extracted_data,
        ) = pickle.load(open("{}_{}.pkl".format(dataset, split), "rb"))
        data_reps = {}
        for id in ids:
            dialogue_know_rep = []
            for sen in extracted_data[id]:
                sen_know_rep = []
                for item in sen:
                    sentence = triple_to_sen(item, relation_dict)
                    # print(sentence)
                    k = tokenizer(sentence, return_tensors="pt")["input_ids"]
                    if cuda:
                        k = k.cuda()
                    embeddings = encoder(k)
                    results = embeddings.last_hidden_state.squeeze(0)[0, :]
                    sen_know_rep.append(results.detach().cpu().numpy().tolist())
                if len(sen_know_rep) == 0:
                    sen_know_rep.append(none_rep)
                dialogue_know_rep.append(sen_know_rep)
                num += 1
                if num % 20 == 0:
                    print(num)
            data_reps[id] = dialogue_know_rep
        pickle.dump(
            [target_context, speakers, cause_labels, emotions, ids, data_reps],
            open("{}_{}_reps".format(dataset, split) + ".pkl", "wb+"),
        )
    else:
        data = pickle.load(open("{}_{}.pkl".format(dataset, split), "rb"))
        for dialogue in data:
            new_dialogue = []
            for sen in dialogue:
                new_sen = {}
                new_sen["text"] = sen["text"]
                new_sen["label"] = sen["label"]
                new_sen["speaker"] = sen["speaker"]
                new_sen["knowledge_representation"] = []
                for item in sen["knowledge"]:
                    sentence = triple_to_sen(item[0], relation_dict)
                    # print(sentence)
                    k = tokenizer(sentence, return_tensors="pt")["input_ids"]
                    if cuda:
                        k = k.cuda()
                    embeddings = encoder(k)
                    results = embeddings.last_hidden_state.squeeze(0)[0, :]
                    new_sen["knowledge_representation"].append(
                        results.detach().cpu().numpy().tolist()
                    )
                # print(new_sen['knowledge_representation'])
                if len(new_sen["knowledge_representation"]) == 0:
                    new_sen["knowledge_representation"].append(none_rep)
                new_dialogue.append(new_sen)
                num += 1
                if num % 20 == 0:
                    print(num)
            new_data.append(new_dialogue)
        pickle.dump(new_data, open("{}_{}_reps.pkl".format(dataset, split), "wb+"))
    # pickle.dump(relation_dict, open('relation_convert_dict.pkl', 'wb+'))
    # a=pickle.load(open('IEMOCAP_val.pkl', 'rb'))
    # print(a)
    # add_relations('RECCON', 'train')
