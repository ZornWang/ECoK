"""utility and helper functions / classes."""
import torch
import json
import pickle
import os
import copy
import logging
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np
import random
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def replace_for_robust_eval(data, rate, num_class):
    """Replace the labels of training data to evaluate robustness."""
    num = len(data)
    for i in range(int(num * rate)):
        data[i]["label"] = random.randint(0, num_class - 1)
    random.shuffle(data)
    return data


def get_num_classes(DATASET: str) -> int:
    """Get the number of classes to be classified by dataset."""
    if DATASET == "MELD":
        NUM_CLASSES = 7
    elif DATASET == "IEMOCAP":
        NUM_CLASSES = 6
    elif DATASET == "DailyDialog":
        NUM_CLASSES = 7
    elif DATASET == "EmoryNLP":
        NUM_CLASSES = 7
    elif DATASET == "RECCON":
        NUM_CLASSES = 7
    else:
        raise ValueError

    return NUM_CLASSES


def get_labels(DATASET: str):
    """Get labels of each dataset."""
    emotions = {}
    assert DATASET in ["MELD", "IEMOCAP", "DailyDialog", "EmoryNLP"]
    # MELD has 7 classes
    emotions["MELD"] = [
        "neutral",
        "joy",
        "surprise",
        "anger",
        "sadness",
        "disgust",
        "fear",
    ]

    # IEMOCAP originally has 11 classes but we'll only use 6 of them.
    emotions["IEMOCAP"] = [
        "neutral",
        "frustration",
        "sadness",
        "anger",
        "excited",
        "happiness",
    ]

    emotions["DailyDialog"] = [
        "none",
        "anger",
        "disgust",
        "fear",
        "happiness",
        "sadness",
        "surprise",
    ]

    emotions["EmoryNLP"] = [
        "Joyful",
        "Neutral",
        "Powerful",
        "Mad",
        "Sad",
        "Scared",
        "Peaceful",
    ]

    return emotions[DATASET]


def compute_VAD_pearson_correlation(predicts, labels):
    pcv = np.corrcoef(predicts[:, 0], labels[:, 0])
    pca = np.corrcoef(predicts[:, 1], labels[:, 1])
    pcd = np.corrcoef(predicts[:, 2], labels[:, 2])
    return pcv, pca, pcd


def get_label_VAD(DATASET: str):
    """Get VAD score of each label from the lexicon NRC-VAD."""
    emotions = get_labels(DATASET)
    label_VAD = []
    VADs = {}
    with open("./utils/NRC-VAD-Lexicon.txt", "r") as f:
        f.readline()
        for line in f:
            scores = line.split()
            key = scores[0] if len(scores) == 4 else " ".join(scores[: len(scores) - 3])
            VADs[key] = torch.FloatTensor(
                [float(scores[-3]), float(scores[-2]), float(scores[-1])]
            )
            # VADs[key] = [float(scores[-3]), float(scores[-2]), float(scores[-1])]
    for i, emotion in enumerate(emotions):
        if emotion == "none":
            label_VAD.append(VADs["Neutral".lower()])
        else:
            label_VAD.append(VADs[emotion.lower()])
    return label_VAD


def convert_label_to_VAD(labels, labels_VAD):
    """Convert labels to corresponding VAD scores."""
    new_labels = []
    for label in labels:
        new_labels.append(labels_VAD[label])
    return torch.stack(new_labels, dim=0)


def compute_predicts(predicts, labels_VAD):
    """Predict categorical emotions from VADs."""
    f_p = []
    for i in range(predicts.shape[0]):
        f = []
        for vad in labels_VAD:
            f.append(float(torch.sum((predicts[i] - vad) ** 2)))
        f_p.append(f.index(min(f)))
    return f_p


def compute_metrics(eval_predictions) -> dict:
    """Return f1_weighted, f1_micro, and f1_macro scores."""
    predictions, label_ids = eval_predictions
    preds = np.argmax(predictions, axis=1)

    f1_weighted = f1_score(label_ids, preds, average="weighted")
    f1_micro = f1_score(label_ids, preds, average="micro")
    f1_macro = f1_score(label_ids, preds, average="macro")

    return {"f1_weighted": f1_weighted, "f1_micro": f1_micro, "f1_macro": f1_macro}


def set_seed(seed: int) -> None:
    """Set random seed to a fixed value.
       Set everything to be deterministic.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def save_latent_params(latent_params_dict, label_records, save_dir):
    for item in latent_params_dict.keys():
        torch.save(
            torch.cat(latent_params_dict[item], dim=0), save_dir + "/" + item + ".pt"
        )
    pickle.dump(label_records, open(save_dir + "/" + "labels.pt", "wb+"))


def get_emotion2id(DATASET: str) -> dict:
    """Get a dict that converts string class to numbers."""
    emotions = {}
    # MELD has 7 classes
    emotions["MELD"] = [
        "neutral",
        "joy",
        "surprise",
        "anger",
        "sadness",
        "disgust",
        "fear",
    ]

    # IEMOCAP originally has 11 classes but we'll only use 6 of them.
    emotions["IEMOCAP"] = [
        "neutral",
        "frustration",
        "sadness",
        "anger",
        "excited",
        "happiness",
    ]

    emotions["DailyDialog"] = [
        "none",
        "anger",
        "disgust",
        "fear",
        "happiness",
        "sadness",
        "surprise",
    ]

    emotions["EmoryNLP"] = [
        "Neutral",
        "Joyful",
        "Powerful",
        "Mad",
        "Sad",
        "Scared",
        "Peaceful",
    ]

    emotions["RECCON"] = [
        "happiness",
        "neutral",
        "anger",
        "sadness",
        "fear",
        "surprise",
        "disgust",
    ]

    emotion2id = {
        DATASET: {emotion: idx for idx, emotion in enumerate(emotions_)}
        for DATASET, emotions_ in emotions.items()
    }

    return emotion2id[DATASET]


class ErcTextDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        DATASET="MELD",
        SPLIT="train",
        speaker_mode="upper",
        num_past_utterances=10,
        num_future_utterances=10,
        model_checkpoint="roberta-base",
        ROOT_DIR="./bart_comet_enhanced_data",
        ONLY_UPTO=False,
        SEED=0,
        COMET_WIN_SIZE=5,
    ):
        """Initialize emotion recognition in conversation text modality dataset class."""

        self.DATASET = DATASET
        self.ROOT_DIR = ROOT_DIR
        self.SPLIT = SPLIT
        self.speaker_mode = speaker_mode
        self.num_past_utterances = num_past_utterances
        self.num_future_utterances = num_future_utterances
        self.model_checkpoint = model_checkpoint
        self.emotion2id = get_emotion2id(self.DATASET)
        self.ONLY_UPTO = ONLY_UPTO
        self.SEED = SEED
        self.COMET_WIN_SIZE = COMET_WIN_SIZE
        self.comet_index = torch.LongTensor([i for i in range(7)])
        self.conceptnet_max_num = 20
        if "roberta" in model_checkpoint:
            self.sep_token = "</s></s>"
        else:
            self.sep_token = "[SEP]"

        set_seed(self.SEED)
        self._load_data()

    def _load_emotions(self):
        """Load the supervised labels"""
        with open(
            os.path.join(self.ROOT_DIR, self.DATASET, "emotions.json"), "r"
        ) as stream:
            self.emotions = json.load(stream)[self.SPLIT]

    def load_utterance_speaker_emotion(self, utt, speaker_mode):
        """Load speaker information for EmoryNLP dataset."""
        utterance = utt["text"].strip()
        emotion = utt["label"]
        if self.DATASET == "DailyDialog":
            speaker = {"A": "Mary", "B": "James"}
            if speaker_mode is not None and speaker_mode.lower() == "upper":
                utterance = speaker[utt["speaker"]].upper() + ": " + utterance
            elif speaker_mode is not None and speaker_mode.lower() == "title":
                utterance = speaker[utt["speaker"]].title() + ": " + utterance
        elif self.DATASET == "EmoryNLP":
            if speaker_mode is not None and speaker_mode.lower() == "upper":
                utterance = utt["speaker"].split()[0].upper() + ": " + utterance
            elif speaker_mode is not None and speaker_mode.lower() == "title":
                utterance = utt["speaker"].split()[0].title() + ": " + utterance
        elif self.DATASET == "IEMOCAP" or self.DATASET == "MELD":
            if speaker_mode is not None and speaker_mode.lower() == "upper":
                utterance = utt["speaker"].upper() + ": " + utterance
            elif speaker_mode is not None and speaker_mode.lower() == "title":
                utterance = utt["speaker"].title() + ": " + utterance
        else:
            raise ValueError(f"{self.DATASET} not supported!!!!!!")

        if "conceptnet" in self.ROOT_DIR:
            reps = utt["knowledge_representation"]
            if len(reps) > self.conceptnet_max_num:
                reps = reps[: self.conceptnet_max_num]
            reps = torch.Tensor(reps)
        else:
            reps = torch.from_numpy(utt["user_features"])

        return {"Utterance": utterance, "Emotion": emotion, "comet_features": reps}

    def _load_data(self):
        """Load data for EmoryNLP dataset."""
        tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint, use_fast=True)
        max_model_input_size = tokenizer.max_model_input_sizes[self.model_checkpoint]
        num_truncated = 0
        inputs = []
        with open(
            os.path.join(self.ROOT_DIR, self.DATASET, self.SPLIT + ".pkl"), "rb"
        ) as f:
            raw_data = pickle.load(f)
            for dialogue in raw_data:
                ues = [
                    self.load_utterance_speaker_emotion(utt, self.speaker_mode)
                    for utt in dialogue
                ]
                num_tokens = [
                    len(tokenizer(ue["Utterance"])["input_ids"]) for ue in ues
                ]
                for idx, ue in enumerate(ues):
                    if ue["Emotion"] not in list(self.emotion2id.keys()):
                        continue

                    label = self.emotion2id[ue["Emotion"]]

                    indexes = [idx]
                    indexes_past = [
                        i
                        for i in range(idx - 1, idx - self.num_past_utterances - 1, -1)
                    ]
                    indexes_future = [
                        i
                        for i in range(idx + 1, idx + self.num_future_utterances + 1, 1)
                    ]

                    offset = 0
                    if len(indexes_past) < len(indexes_future):
                        for _ in range(len(indexes_future) - len(indexes_past)):
                            indexes_past.append(None)
                    elif len(indexes_past) > len(indexes_future):
                        for _ in range(len(indexes_past) - len(indexes_future)):
                            indexes_future.append(None)

                    for i, j in zip(indexes_past, indexes_future):
                        if i is not None and i >= 0:
                            indexes.insert(0, i)
                            offset += 1
                            if (
                                sum([num_tokens[idx_] + 2 for idx_ in indexes]) - 2
                                > max_model_input_size
                            ):
                                del indexes[0]
                                offset -= 1
                                num_truncated += 1
                                break
                        if j is not None and j < len(ues):
                            indexes.append(j)
                            if (
                                sum([num_tokens[idx_] + 2 for idx_ in indexes]) - 2
                                > max_model_input_size
                            ):
                                del indexes[-1]
                                num_truncated += 1
                                break

                    utterances = [ues[idx_]["Utterance"] for idx_ in indexes]

                    if "conceptnet" in self.ROOT_DIR:
                        comet_features = pad_sequence(
                            [ues[idx_]["comet_features"] for idx_ in indexes],
                            batch_first=True,
                            padding_value=0.0,
                        )
                        if comet_features.shape[1] < self.conceptnet_max_num:
                            comet_features = torch.cat(
                                [
                                    comet_features,
                                    torch.zeros(
                                        [
                                            comet_features.shape[0],
                                            self.conceptnet_max_num
                                            - comet_features.shape[1],
                                            comet_features.shape[2],
                                        ]
                                    ),
                                ],
                                dim=1,
                            )
                    else:
                        comet_features = torch.stack(
                            [ues[idx_]["comet_features"] for idx_ in indexes], dim=0
                        )
                        comet_features = torch.index_select(
                            comet_features, 1, self.comet_index
                        )

                    all_sent_labels = [
                        self.emotion2id[ues[idx_]["Emotion"]]
                        if ues[idx_]["Emotion"] in list(self.emotion2id.keys())
                        else 0
                        for idx_ in indexes
                    ]
                    comet_features, final_pos, comet_pos = self.prepare_comet_features(
                        comet_features, all_sent_labels, offset
                    )
                    if len(utterances) == 1:
                        final_part_2 = utterances[offset] + self.sep_token
                        current_utt = utterances[offset]

                        input_ids_attention_mask_part_2 = tokenizer(final_part_2)
                        input_ids = input_ids_attention_mask_part_2["input_ids"]
                        attention_mask = [1] * len(input_ids)
                        if self.num_future_utterances == 0:
                            utt_pos_masks = (
                                [[0] * len(input_ids)] * final_pos[0]
                                + [attention_mask]
                                + [[0] * len(input_ids)]
                                * (self.COMET_WIN_SIZE + 1 - final_pos[1])
                            )
                        else:
                            utt_pos_masks = (
                                [[0] * len(input_ids)] * final_pos[0]
                                + [attention_mask]
                                + [[0] * len(input_ids)]
                                * (2 * self.COMET_WIN_SIZE + 1 - final_pos[1])
                            )
                    else:
                        utt_pos_masks = []
                        for cuu_offset in range(comet_pos[0], comet_pos[1]):
                            if cuu_offset != 0:
                                part_1 = (
                                    self.sep_token.join(utterances[:cuu_offset])
                                    + self.sep_token
                                )
                            else:
                                part_1 = self.sep_token.join(utterances[:cuu_offset])
                            if cuu_offset < len(utterances) - 1:
                                part_2 = utterances[cuu_offset] + self.sep_token
                            else:
                                part_2 = utterances[cuu_offset]
                            part_3 = self.sep_token.join(utterances[cuu_offset + 1 :])

                            input_ids_attention_mask_part_1 = tokenizer(part_1)
                            input_ids_attention_mask_part_2 = tokenizer(part_2)
                            input_ids_attention_mask_part_3 = tokenizer(part_3)

                            cuu_mask = (
                                [0]
                                * (
                                    len(
                                        input_ids_attention_mask_part_1[
                                            "attention_mask"
                                        ]
                                    )
                                    - 1
                                )
                                + input_ids_attention_mask_part_2["attention_mask"][
                                    1:-1
                                ]
                                + [0]
                                * (
                                    len(
                                        input_ids_attention_mask_part_3[
                                            "attention_mask"
                                        ]
                                    )
                                    - 1
                                )
                            )
                            utt_pos_masks.append(cuu_mask)
                            # print(input_ids_attention_mask_part_3['input_ids'])
                            # print(input_ids_attention_mask_part_3['attention_mask'])
                            if cuu_offset == offset:
                                current_utt = utterances[cuu_offset]
                                input_ids = (
                                    input_ids_attention_mask_part_1["input_ids"][:-1]
                                    + input_ids_attention_mask_part_2["input_ids"][1:-1]
                                    + input_ids_attention_mask_part_3["input_ids"][1:]
                                )
                                attention_mask = [1] * len(input_ids)
                        if self.num_future_utterances == 0:
                            utt_pos_masks = (
                                [[0] * len(input_ids)] * final_pos[0]
                                + utt_pos_masks
                                + [[0] * len(input_ids)]
                                * (self.COMET_WIN_SIZE + 1 - final_pos[1])
                            )
                        else:
                            utt_pos_masks = (
                                [[0] * len(input_ids)] * final_pos[0]
                                + utt_pos_masks
                                + [[0] * len(input_ids)]
                                * (2 * self.COMET_WIN_SIZE + 1 - final_pos[1])
                            )

                    for k in utt_pos_masks:
                        assert len(k) == len(input_ids)
                    if self.num_future_utterances == 0:
                        assert self.COMET_WIN_SIZE + 1 == len(utt_pos_masks)
                    else:
                        assert 2 * self.COMET_WIN_SIZE + 1 == len(utt_pos_masks)

                    current_utt_mask = tokenizer(current_utt)
                    current_ids = current_utt_mask["input_ids"]
                    current_masks = current_utt_mask["attention_mask"]
                    input_ = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "pos_masks": utt_pos_masks,
                        "comet_features": comet_features,
                        "label": label,
                        "target utterance": {
                            "ids": current_ids,
                            "masks": current_masks,
                        },
                    }
                    inputs.append(input_)
            logging.info(f"number of truncated utterances: {num_truncated}")
            self.inputs_ = inputs

    def prepare_comet_features(self, comet_feature, all_sent_labels, off_set):
        if self.num_future_utterances == 0:
            comet_sent_labels = [-1] * (self.COMET_WIN_SIZE + 1)
            final_features = torch.zeros(
                [
                    self.COMET_WIN_SIZE + 1,
                    comet_feature.shape[1],
                    comet_feature.shape[2],
                ]
            )
            comet_mask = torch.ones(self.COMET_WIN_SIZE + 1)
            comet_left = off_set - self.COMET_WIN_SIZE
            comet_right = off_set + 1
            final_left = 0
            final_right = self.COMET_WIN_SIZE + 1
            if off_set < self.COMET_WIN_SIZE:
                comet_mask[0 : self.COMET_WIN_SIZE - off_set] = 0.0
                comet_left = 0
                final_left = self.COMET_WIN_SIZE - off_set
        else:
            comet_sent_labels = [-1] * (2 * self.COMET_WIN_SIZE + 1)
            final_features = torch.zeros(
                [
                    2 * self.COMET_WIN_SIZE + 1,
                    comet_feature.shape[1],
                    comet_feature.shape[2],
                ]
            )
            comet_mask = torch.ones(2 * self.COMET_WIN_SIZE + 1)
            comet_left = off_set - self.COMET_WIN_SIZE
            comet_right = off_set + self.COMET_WIN_SIZE + 1
            final_left = 0
            final_right = 2 * self.COMET_WIN_SIZE + 1
            if off_set < self.COMET_WIN_SIZE:
                comet_mask[0 : self.COMET_WIN_SIZE - off_set] = 0.0
                comet_left = 0
                final_left = self.COMET_WIN_SIZE - off_set
            if comet_feature.shape[0] < off_set + self.COMET_WIN_SIZE + 1:
                comet_mask[
                    self.COMET_WIN_SIZE
                    - off_set
                    + comet_feature.shape[0] : 2 * self.COMET_WIN_SIZE
                    + 1
                ] = 0.0
                comet_right = comet_feature.shape[0]
                final_right = self.COMET_WIN_SIZE - off_set + comet_feature.shape[0]
        final_features[final_left:final_right,] = comet_feature[
            comet_left:comet_right,
        ]
        comet_sent_labels[final_left:final_right] = all_sent_labels[
            comet_left:comet_right
        ]
        return (
            {
                "feature": final_features,
                "mask": comet_mask,
                "comet_sent_labels": comet_sent_labels,
            },
            [final_left, final_right],
            [comet_left, comet_right],
        )

    def __len__(self):
        return len(self.inputs_)

    def __getitem__(self, index):

        return self.inputs_[index]


class RECCONTextDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        DATASET="RECCON",
        SPLIT="train",
        speaker_mode="upper",
        model_checkpoint="roberta-base",
        ROOT_DIR="./comet_enhanced_data",
        ONLY_UPTO=False,
        SEED=0,
    ):
        """Initialize emotion casual entailment text modality dataset class."""

        self.DATASET = DATASET
        self.ROOT_DIR = ROOT_DIR
        self.SPLIT = SPLIT
        self.speaker_mode = speaker_mode
        self.model_checkpoint = model_checkpoint
        self.emotion2id = get_emotion2id(self.DATASET)
        self.ONLY_UPTO = ONLY_UPTO
        self.SEED = SEED
        self.conceptnet_max_num = 15
        if "roberta" in model_checkpoint:
            self.sep_token = "</s></s>"
        else:
            self.sep_token = "[SEP]"

        set_seed(self.SEED)
        self._load_data()

    def _load_emotions(self):
        """Load the supervised labels"""
        with open(
            os.path.join(self.ROOT_DIR, self.DATASET, "emotions.json"), "r"
        ) as stream:
            self.emotions = json.load(stream)[self.SPLIT]

    def load_utterance_speaker(self, utts, speakers, speaker_mode):
        """Load speaker information for EmoryNLP dataset."""
        utterances = []
        for utterance, speaker in zip(utts, speakers):
            trans_speaker = {"A": "Mary", "B": "James"}
            if speaker_mode is not None and speaker_mode.lower() == "upper":
                cu_speaker = trans_speaker[speaker].upper()
            elif speaker_mode is not None and speaker_mode.lower() == "title":
                cu_speaker = trans_speaker[speaker].title()
            utterance = cu_speaker + ": " + utterance
            utterances.append(utterance)
        return utterances

    def _load_data(self):
        """Load data for EmoryNLP dataset."""
        tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint, use_fast=True)
        max_model_input_size = tokenizer.max_model_input_sizes[self.model_checkpoint]
        num_truncated = 0
        inputs = []
        with open(
            os.path.join(self.ROOT_DIR, self.DATASET, self.SPLIT + ".pkl"), "rb"
        ) as f:
            (
                target_context,
                speakers,
                cause_labels,
                emotions,
                ids,
                extracted_data,
            ) = pickle.load(f)
            for id in ids:
                dialogue = target_context[id]
                speaker = speakers[id]
                cause_label = cause_labels[id]
                emotion = emotions[id]
                comet_data = extracted_data[id]
                ues = self.load_utterance_speaker(dialogue, speaker, self.speaker_mode)
                ue = self.sep_token.join(ues)
                k = tokenizer(ue)
                tokenized_utt = k["input_ids"]
                input_mask = k["attention_mask"]
                length = len(tokenized_utt)

                start = 1
                while length > max_model_input_size:
                    ue = self.sep_token.join(ues[start:])
                    k = tokenizer(ue)
                    tokenized_utt = k["input_ids"]
                    input_mask = k["attention_mask"]
                    length = len(tokenized_utt)

                    cause_label = cause_label[1:]
                    emotion = emotion[1:]
                    comet_data = comet_data[1:]

                    start += 1
                assert len(ues) - start + 1 == len(cause_label)
                utt_masks = torch.zeros([len(cause_label), len(tokenized_utt)])
                cuu_utt = 0
                for i in range(1, len(tokenized_utt)):
                    utt_masks[cuu_utt, i] = 1
                    if tokenized_utt[i] == tokenized_utt[i - 1] == 2:
                        cuu_utt += 1
                assert cuu_utt + 1 == len(cause_label)
                if "conceptnet" in self.ROOT_DIR:
                    new_comet_data = []
                    for item in comet_data:
                        if len(item) >= self.conceptnet_max_num:
                            new_comet_data.append(item[: self.conceptnet_max_num])
                        else:
                            new_comet_data.append(
                                item
                                + [[0.0] * len(item[0])]
                                * (self.conceptnet_max_num - len(item))
                            )
                else:
                    new_comet_data = comet_data
                input_ = {
                    "input_ids": tokenized_utt,
                    "attention_mask": input_mask,
                    "pos_masks": utt_masks,
                    "comet_features": new_comet_data,
                    "label": cause_label,
                    "emolabel": emotion,
                }
                inputs.append(input_)

            logging.info(f"number of truncated utterances: {num_truncated}")
            self.inputs_ = inputs

    def __len__(self):
        return len(self.inputs_)

    def __getitem__(self, index):

        return self.inputs_[index]


class DD(dict):
    def __getattr__(self, attr):
        if attr == "__getstate__":
            return super(DD, self).__getstate__
        elif attr == "__setstate__":
            return super(DD, self).__setstate__
        elif attr == "__slots__":
            return super(DD, self).__slots__
        return self[attr]

    def __setattr__(self, attr, value):
        # Safety check to ensure consistent behavior with __getattr__.
        assert attr not in ("__getstate__", "__setstate__", "__slots__")
        #         if attr.startswith('__'):
        #             return super(DD, self).__setattr__(attr, value)
        self[attr] = value

    def __str__(self):
        return "DD%s" % dict(self)

    def __repr__(self):
        return str(self)

    def __deepcopy__(self, memo):
        z = DD()
        for k, kv in self.items():
            z[k] = copy.deepcopy(kv, memo)
        return z


if __name__ == "__main__":
    """ds_train = ErcTextDataset(DATASET='MELD', SPLIT='val',
                                 model_checkpoint='roberta-base',
                                 ROOT_DIR="../conceptnet_enhanced_data", SEED=42)"""
    ds_train = RECCONTextDataset(
        DATASET="RECCON",
        SPLIT="val",
        model_checkpoint="roberta-base",
        ROOT_DIR="../conceptnet_enhanced_data",
        SEED=42,
    )
    a = ds_train.inputs_
    # replace_for_robust_eval(a, 0.05, 6)
    # print(prompt_get_emotion2id('MELD'))
