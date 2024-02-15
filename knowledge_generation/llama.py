import torch
import timeit
import json
import argparse
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig
from flask import Flask, request

import os

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

parser = argparse.ArgumentParser(description="模型大小")
parser.add_argument("--model", type=str, default="13B", help="模型大小")
args = parser.parse_args()

app = Flask(__name__)

generation_config = GenerationConfig(
    max_new_tokens=2048,
    num_return_sequences=1,
    repetition_penalty=1.2,
    # temperature=0.1,
    top_p=0.75,
    top_k=40,
    # num_beams = 4,
    do_sample=True,
)

model_list = {
    "13B": "meta-llama/Llama-2-13b-chat-hf",
    "70B": "meta-llama/Llama-2-70b-chat-hf",
}
model_scale = args.model
model_name = model_list[model_scale]
access_token = "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"

if model_scale == "70B":
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        load_in_4bit=True,
        torch_dtype=torch.float16,
        token=access_token,
    )
elif model_scale == "13B":
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        token=access_token,
    )
tokenizer = LlamaTokenizer.from_pretrained(
    model_name,
    token=access_token,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token

model.eval()


def get_tokens_as_list(word_list):
    "Converts a sequence of words into a list of tokens"
    tokenizer_with_prefix_space = LlamaTokenizer.from_pretrained(
        model_name, token=access_token, add_prefix_space=True
    )
    if tokenizer_with_prefix_space.pad_token is None:
        tokenizer_with_prefix_space.pad_token = tokenizer_with_prefix_space.unk_token

    tokens_list = []
    for word in word_list:
        tokenized_word = tokenizer_with_prefix_space(
            [word], add_special_tokens=False
        ).input_ids[0]
        tokens_list.append(tokenized_word)
    return tokens_list


@app.route("/chat", methods=["POST"])
def chat():
    data = json.loads(request.get_data(as_text=True))
    text = data["text"]
    start = timeit.default_timer()
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    if "bad_words" in data:
        bad_words = data["bad_words"]
        bad_words_ids = get_tokens_as_list(word_list=bad_words)
        generation_config.bad_words_ids = bad_words_ids
    with torch.no_grad():
        output_ids = model.generate(
            inputs.input_ids,
            generation_config=generation_config,
            # eos_token_id=tokenizer.eos_token_id,
        )
    output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    end = timeit.default_timer()
    return json.dumps(
        {"response": output_text, "time": end - start}, ensure_ascii=False, indent=4
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
