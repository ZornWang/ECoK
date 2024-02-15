from tqdm import tqdm
from nltk import tokenize
import numpy as np
import pickle, torch
import comet.src.data.data as data
import comet.src.data.config as cfg
import comet.src.models.utils as model_utils
import comet.src.interactive.functions as interactive


class CSKFeatureExtractor:
    def __init__(self):
        super(CSKFeatureExtractor, self).__init__()

        device = "cpu"
        # device = 0
        model_file = "comet/pretrained_models/atomic_pretrained_model.pickle"
        sampling_algorithm = "beam-5"
        category = "all"

        opt, state_dict = interactive.load_model_file(model_file)
        data_loader, text_encoder = interactive.load_data("atomic", opt)

        self.opt = opt
        self.data_loader = data_loader
        self.text_encoder = text_encoder

        n_ctx = data_loader.max_event + data_loader.max_effect
        n_vocab = len(text_encoder.encoder) + n_ctx
        self.model = interactive.make_model(opt, n_vocab, n_ctx, state_dict)
        self.model.eval()

        if device != "cpu":
            cfg.device = int(device)
            cfg.do_gpu = True
            torch.cuda.set_device(cfg.device)
            self.model.cuda(cfg.device)
        else:
            cfg.device = "cpu"

    def set_atomic_inputs(self, input_event, category, data_loader, text_encoder):
        XMB = torch.zeros(1, data_loader.max_event + 1).long().to(cfg.device)
        prefix, suffix = data.atomic_data.do_example(
            text_encoder, input_event, None, True, None
        )

        if len(prefix) > data_loader.max_event + 1:
            prefix = prefix[: data_loader.max_event + 1]

        XMB[:, : len(prefix)] = torch.LongTensor(prefix)
        XMB[:, -1] = torch.LongTensor([text_encoder.encoder["<{}>".format(category)]])

        batch = {}
        batch["sequences"] = XMB
        batch["attention_mask"] = data.atomic_data.make_attention_mask(XMB)
        return batch

    def extract(self, sentences):
        atomic_keys = [
            "xIntent",
            "xAttr",
            "xNeed",
            "xWant",
            "xEffect",
            "xReact",
            "oWant",
            "oEffect",
            "oReact",
        ]
        features = []

        for sent in sentences:
            seqs = []
            masks = []
            for category in atomic_keys:
                batch = self.set_atomic_inputs(
                    sent, category, self.data_loader, self.text_encoder
                )
                seqs.append(batch["sequences"])
                masks.append(batch["attention_mask"])

            XMB = torch.cat(seqs)
            MMB = torch.cat(masks)
            XMB = model_utils.prepare_position_embeddings(
                self.opt, self.data_loader.vocab_encoder, XMB.unsqueeze(-1)
            )
            h, _ = self.model(XMB.unsqueeze(1), sequence_mask=MMB)
            feature = h[:, -1, :].detach().cpu().numpy()
            features.append(feature)
            # last_index = MMB[0][:-1].nonzero()[-1].cpu().numpy()[0] + 1

        return features
