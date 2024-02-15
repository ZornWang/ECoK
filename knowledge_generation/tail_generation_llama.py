import os
import re
import requests
import argparse
import json
import jsonlines
from tqdm import tqdm
from retrying import retry
from collections import OrderedDict
import random


def load_seed(seed):
    """Function to set random seed."""
    random.seed(seed)


def response_process(text):
    """
    提取json字符串，如果json字符串不是以}结尾，则加上}，并且去掉换行
    :param text: 原始字符串
    :return: json字符串
    """
    text = text[text.find("{") : text.rfind("}") + 1]
    text.replace("\n", "")
    if text[-1] != "}" and text[0] == "{":
        text += "}"
    return text


def process_llama_output(output):
    """
    处理llama模型输出
    :param output: llama输出
    :return: 处理后数据
    """
    result = output[output.rfind("[/INST]") + len("[/INST]") :]
    return result


@retry(wait_fixed=1000, stop_max_attempt_number=3)
def llama(prompt, verbose: bool = False):
    url = "http://localhost:10000/chat"
    data = json.dumps({"text": prompt})
    response = requests.post(url, data=data)
    response = json.loads(response.text)
    result = {}
    result["response"] = process_llama_output(response["response"])
    if verbose:
        print("Response: ", result["response"])
    return result


def main():
    """
    Script to generate and save Llama generations.
    """
    load_seed(84)

    parser = argparse.ArgumentParser(description="Arguments for Llama generation.")
    parser.add_argument(
        "--incontext_examples_path",
        type=str,
        default=os.path.join("data", "raw", "json", "train.json"),
        help="",
    )
    parser.add_argument(
        "--llama_inputs_path",
        type=str,
        default=os.path.join("data", "raw", "json", "test.json"),
        help="Path for json file of inputs to generate new Llama tails.",
    )
    parser.add_argument(
        "--llama_dict_path",
        type=str,
        default=os.path.join("data", "generate", "llama", "llama70_output_dicts.jsonl"),
        help="Path for jsonlines file to save Llama output dictionaries.",
    )
    parser.add_argument(
        "--llama_text_path",
        type=str,
        default=os.path.join("data", "generate", "llama", "llama70_kg_data.json"),
        help="Path for json file (of head/rel/tails) to save new Llama tails.",
    )
    parser.add_argument(
        "--llama_top1_text_path",
        type=str,
        default=os.path.join("data", "generate", "llama", "llama70_top1picks.json"),
        help="Path for json file (of head/rel/tails) to only save best Llama tails according to OpenAI.",
    )
    parser.add_argument("--k_shot", type=int, default=5, help="Few-shot k.")
    parser.add_argument(
        "--verbose", action="store_true", help="Flag to make script verbose."
    )
    args = parser.parse_args()

    # 1) Load the in-context examples and gather them by head and rel, save as dict
    with open(args.incontext_examples_path, mode="r", encoding="utf-8") as f:
        incontext_examples = json.load(f)
        f.close()

    rel_to_head_to_tail_dict = OrderedDict()
    for str_instance in incontext_examples:
        str_list = str_instance.split("\t")
        head = str_list[0]
        rel = str_list[1]
        tail = str_list[2]
        head_rel_key = "{}\t{}".format(head, rel)
        if rel not in rel_to_head_to_tail_dict.keys():
            rel_to_head_to_tail_dict[rel] = OrderedDict()
        if head not in rel_to_head_to_tail_dict[rel].keys():
            rel_to_head_to_tail_dict[rel][head] = tail

    # 2) Load the tuples and separate head/rels that we need to generate for
    with open(args.llama_inputs_path, mode="r", encoding="utf-8") as f:
        input_tuples = json.load(f)
        f.close()
    input_queries = set()
    for str_instance in input_tuples:
        str_list = str_instance.split("\t")
        head = str_list[0]
        rel = str_list[1]
        head_rel_key = "{}\t{}".format(head, rel)
        input_queries.add(head_rel_key)
    input_queries = list(input_queries)
    input_queries.sort()
    print("Generating for {} queries...".format(len(input_queries)))

    # NOTE: uncomment if testing the script // only 3 generations
    #       so we don't request too much at once from OpenAI API
    input_queries = random.sample(population=input_queries, k=150)

    # 3) Request generation from LLMs
    llama_gen_text = []
    llama_gen_dict_list = []
    query_loop = tqdm(input_queries, desc="generation: ")

    for input_query in query_loop:
        # a) get the context
        if args.verbose:
            print("-" * 50)
            print("Input query: ", input_query)
        str_list = input_query.split("\t")
        head = str_list[0]
        rel = str_list[1]
        context_heads = []
        contexts = []
        input_prompt = ""
        # b) attach to input query as a prompt
        if args.k_shot == 0:
            with open(os.path.join("resources", "zero-shot.txt"), "r") as file:
                prompt_file = file.read()
                file.close()
            input = json.dumps({"head": head, "relations": rel})
            input_prompt = re.sub(r"\{\{ input \}\}", str(input), prompt_file)
        else:
            context_heads = [
                h for h in rel_to_head_to_tail_dict[rel].keys() if h != head
            ]
            context_heads = random.sample(population=context_heads, k=args.k_shot)
            if args.verbose:
                print("Sampled context heads: ", context_heads)
            with open(os.path.join("resources", "5-shot.txt"), "r") as file:
                prompt_file = file.read()
                file.close()
            contexts = [
                json.dumps(
                    {
                        "head": h,
                        "relations": rel,
                        "tails": [rel_to_head_to_tail_dict[rel][h]],
                    }
                )
                for h in context_heads
            ]
            input_prompt = re.sub(
                r"\{\{ examples \}\}", str(("\n").join(contexts)), prompt_file
            )
            input = json.dumps({"head": head, "relations": rel})
            input_prompt = re.sub(r"\{\{ input \}\}", str(input), input_prompt)
            # contexts = [h + "\t" + rel + "\t" + rel_to_head_to_tail_dict[rel][h] + "\n\n" for h in context_heads]
            # input_prompt = ('').join(contexts) + head  + "\t" + rel + "\t"

        if args.verbose:
            print("Input prompt:")
            print(input_prompt)

        # c) get the generation
        try:
            result = llama(input_prompt, args.verbose)
            result["input_prompt"] = input_prompt
            result["context_heads"] = context_heads
            result["contexts"] = contexts
            result["head"] = head
            result["rel"] = rel
            response = result["response"]
            text_json = json.loads(response_process(response))
            tails = text_json["tails"]
            assert len(tails) == 5
            result["tails"] = tails

            llama_gen_dict_list.append(result)
            llama_gen_text.extend([head + "\t" + rel + "\t" + tail for tail in tails])
        except Exception as e:
            print(e)
            continue
        # d) save the output dicts and the data in head \t rel \t tail format
        with open(args.llama_dict_path, mode="w", encoding="utf-8") as f:
            writer = jsonlines.Writer(f)
            writer.write_all(llama_gen_dict_list)
        writer.close()
        f.close()

        with open(args.llama_text_path, mode="w", encoding="utf-8") as f:
            json.dump(llama_gen_text, f, ensure_ascii=False, indent=4)
            f.close()

    # 4) Pick first instance of head rel only
    seen_head_sel = []
    top1pick_list = []

    for gen_string in llama_gen_text:
        head_rel_key = gen_string.split("\t")[:2]
        if head_rel_key not in seen_head_sel:
            top1pick_list.append(gen_string)
            seen_head_sel.append(head_rel_key)

    # 5) Save this version as Llama's top 1 pick
    with open(args.llama_top1_text_path, mode="w", encoding="utf-8") as f:
        json.dump(top1pick_list, f, ensure_ascii=False, indent=4)
        f.close()

    print(
        "Llama generations saved at {} and {}.".format(
            args.llama_text_path, args.llama_top1_text_path
        )
    )


if __name__ == "__main__":
    main()
