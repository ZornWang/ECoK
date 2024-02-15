import random

import nltk
import nltk.data
import pickle
import numpy as np
import csv
import os
import numpy
import pandas as pd
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from csk_feature_extract import CSKFeatureExtractor


class ErcDatasetComet(torch.utils.data.Dataset):
    def __init__(self, DATASET="MELD", SPLIT="train"):
        """Initialize emotion recognition in conversation text modality dataset class."""

        self.DATASET = DATASET
        self.SPLIT = SPLIT
        if self.DATASET in ["MELD", "IEMOCAP"]:
            self._load_emotions()
            self._load_utterance_ordered()
            self.data = self._string2tokens()
        else:
            self.data = self._load_data()

    def _load_emotions(self):
        """Load the supervised labels"""
        with open(os.path.join("./data", self.DATASET, "emotions.json"), "r") as stream:
            self.emotions = json.load(stream)[self.SPLIT]

    def _load_data(self):
        """Load data for EmoryNLP dataset."""
        with open(
            os.path.join("./data", self.DATASET, self.SPLIT + "_data.json"), "r"
        ) as f:
            raw_data = json.load(f)
        return raw_data

    def _load_utterance_ordered(self):
        """Load the ids of the utterances in order."""
        with open(
            os.path.join("./data", self.DATASET, "utterance-ordered.json"), "r"
        ) as stream:
            utterance_ordered = json.load(stream)[self.SPLIT]

        count = 0
        self.utterance_ordered = {}
        for diaid, uttids in utterance_ordered.items():
            self.utterance_ordered[diaid] = []
            for uttid in uttids:
                try:
                    with open(
                        os.path.join(
                            "./data",
                            self.DATASET,
                            "raw-texts",
                            self.SPLIT,
                            uttid + ".json",
                        ),
                        "r",
                    ) as stream:
                        foo = json.load(stream)
                    self.utterance_ordered[diaid].append(uttid)
                except Exception as e:
                    count += 1

    def _load_utterance_speaker_emotion(self, uttid) -> dict:
        """Load an speaker-name prepended utterance and emotion label"""
        text_path = os.path.join(
            "./data", self.DATASET, "raw-texts", self.SPLIT, uttid + ".json"
        )

        with open(text_path, "r") as stream:
            text = json.load(stream)

        utterance = text["Utterance"].strip()
        emotion = text["Emotion"]

        if self.DATASET == "MELD":
            speaker = text["Speaker"]
        elif self.DATASET == "IEMOCAP":
            sessid = text["SessionID"]
            # https: // www.ssa.gov/oact/babynames/decades/century.html
            speaker = {
                "Ses01": {"Female": "Mary", "Male": "James"},
                "Ses02": {"Female": "Patricia", "Male": "John"},
                "Ses03": {"Female": "Jennifer", "Male": "Robert"},
                "Ses04": {"Female": "Linda", "Male": "Michael"},
                "Ses05": {"Female": "Elizabeth", "Male": "William"},
            }[sessid][text["Speaker"]]

        else:
            raise ValueError(f"{self.DATASET} not supported!!!!!!")

        return {"text": utterance, "label": emotion, "speaker": speaker}

    def _create_input(self, diaids):
        inputs = []
        for diaid in tqdm(diaids):
            ues = [
                self._load_utterance_speaker_emotion(uttid)
                for uttid in self.utterance_ordered[diaid]
            ]
            inputs.append(ues)
        return inputs

    def _string2tokens(self):
        """Convert string to (BPE) tokens."""
        diaids = sorted(list(self.utterance_ordered.keys()))

        return self._create_input(diaids=diaids)


def comet_atomic_feature_extract(dataset_name, save_dir):
    extractor = CSKFeatureExtractor()

    for part in ["train", "val", "test"]:
        dataset = ErcDatasetComet(dataset_name, part)
        extracted_data = []
        print("Processing for {} set.".format(part))
        for j, dialogue in enumerate(dataset.data):
            inputs = [utt["text"] for utt in dialogue]
            extracted_dialogue = []
            features = extractor.extract(inputs)
            for i, utt in enumerate(dialogue):
                utt["atomic_features"] = features[i]
                extracted_dialogue.append(utt)
            extracted_data.append(extracted_dialogue)
            if j % 100 == 0:
                print("{} dialogues processed.".format(j))
        pickle.dump(
            extracted_data,
            open(
                os.path.join("./comet_enhanced_data", dataset_name, part + ".pkl"),
                "wb+",
            ),
        )


def RECCON_get_csk_feature():
    extractor = CSKFeatureExtractor()
    for split in ["train", "val", "test"]:
        target_context, speaker, cause_label, emotion, ids = pickle.load(
            open("./RECCON_data/dailydialog_" + split + ".pkl", "rb"), encoding="latin1"
        )
        extracted_data = {}
        for j, id in enumerate(ids):
            dialogue = target_context[id]
            features = extractor.extract(dialogue)
            extracted_data[id] = features
            if j % 100 == 0:
                print("{} dialogues processed.".format(j))
        pickle.dump(
            [target_context, speaker, cause_label, emotion, ids, extracted_data],
            open("./RECCON/" + split + ".pkl", "wb"),
        )


if __name__ == "__main__":
    # make_sad_comet_data('./SAD/test.csv', './SAD')
    # make_depression_comet_data("./depression")
    # make_comet_data("./dreaddit/dreaddit-train.csv", None)
    # read_sad_data("./SAD/train.csv")
    RECCON_get_csk_feature()
