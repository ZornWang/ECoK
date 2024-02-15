import os
import argparse
import json
import re
import jsonlines
import openai
from tqdm import tqdm
from collections import OrderedDict
import random


def load_seed(seed):
    """Function to set random seed."""
    random.seed(seed)


def main():
    """
    Script to generate and save GPT generations.
    """
    load_seed(84)

    parser = argparse.ArgumentParser(description="Arguments for GPT generation.")
    parser.add_argument(
        "--incontext_examples_path",
        type=str,
        default=os.path.join("data", "raw", "json", "train.json"),
        help="",
    )
    parser.add_argument(
        "--gpt_inputs_path",
        type=str,
        default=os.path.join("data", "raw", "json", "test.json"),
        help="Path for json file of inputs to generate new GPT tails.",
    )
    parser.add_argument(
        "--gpt_dict_path",
        type=str,
        default=os.path.join("data", "generate", "gpt", "gpt4_output_dicts.jsonl"),
        help="Path for jsonlines file to save GPT output dictionaries.",
    )
    parser.add_argument(
        "--gpt_text_path",
        type=str,
        default=os.path.join("data", "generate", "gpt", "gpt4_kg_data.json"),
        help="Path for json file (of head/rel/tails) to save new GPT tails.",
    )
    parser.add_argument(
        "--gpt_top1_text_path",
        type=str,
        default=os.path.join("data", "generate", "gpt", "gpt4_top1picks.json"),
        help="Path for json file (of head/rel/tails) to only save best GPT tails according to OpenAI.",
    )
    parser.add_argument("--k_shot", type=int, default=5, help="Few-shot k.")
    parser.add_argument(
        "--gpt_model_name",
        type=str,
        default="gpt-4-1106-preview",
        # default="gpt-3.5-turbo-1106",
        help="GPT OpenAI API model name.",
    )
    parser.add_argument(
        "--openai_api_base",
        type=str,
        default="https://api.openai.com/v1",
        help="OpenAI API base.",
    )
    parser.add_argument(
        "--openai_api_key",
        type=str,
        default="sk-**",
        help="OpenAI API key.",
    )
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
    with open(args.gpt_inputs_path, mode="r", encoding="utf-8") as f:
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

    # 3) Request generation from OpenAI
    gpt_gen_text = []
    gpt_gen_dict_list = []
    query_loop = tqdm(input_queries, desc="generation: ")
    # openai.api_key = args.openai_api_key
    client = openai.OpenAI(api_key=args.openai_api_key, base_url=args.openai_api_base)

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

        if args.verbose:
            print("Input prompt:")
            print(input_prompt)

        # c) get the generation from open AI
        completion = client.chat.completions.create(
            model=args.gpt_model_name,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_prompt},
            ],
            # max_tokens=12,
            top_p=1.0,
            n=5,
            temperature=0.9,
        )
        result = {}
        result["input_prompt"] = input_prompt
        result["context_heads"] = context_heads
        result["contexts"] = contexts
        result["head"] = head
        result["rel"] = rel
        try:
            tails = json.loads(completion.choices[0].message.content)["tails"]
            assert len(tails) == 5
            result["tails"] = tails

            gpt_gen_dict_list.append(result)
            gpt_gen_text.extend([head + "\t" + rel + "\t" + tail for tail in tails])
        except Exception as e:
            print(e)
            continue

        # d) save the output dicts and the data in head \t rel \t tail format
        with open(args.gpt_dict_path, mode="w", encoding="utf-8") as f:
            writer = jsonlines.Writer(f)
            writer.write_all(gpt_gen_dict_list)
        writer.close()
        f.close()

        with open(args.gpt_text_path, mode="w", encoding="utf-8") as f:
            json.dump(gpt_gen_text, f, ensure_ascii=False, indent=4)
            f.close()

    # 4) Pick first instance of head rel only
    seen_head_sel = []
    top1pick_list = []

    for gen_string in gpt_gen_text:
        head_rel_key = gen_string.split("\t")[:2]
        if head_rel_key not in seen_head_sel:
            top1pick_list.append(gen_string)
            seen_head_sel.append(head_rel_key)

    # 5) Save this version as GPT's top 1 pick
    with open(args.gpt_top1_text_path, mode="w", encoding="utf-8") as f:
        json.dump(top1pick_list, f, ensure_ascii=False, indent=4)
        f.close()

    print(
        "GPT generations saved at {} and {}.".format(
            args.gpt_text_path, args.gpt_top1_text_path
        )
    )


if __name__ == "__main__":
    main()
