import pickle as pkl
import argparse
from transformers import AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Decode the dumped output ids into text")
    parser.add_argument("--tokenizer_pth", type=str, required=True, help="Path to tokenizer checkpoint")
    parser.add_argument("--dumped_output_ids_path", type=str, required=True, help="Path to dumped output ids data")
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_pth, 
        model_max_length=2048, 
        padding_side="right", 
        use_fast=False
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "right"

    with open(args.dumped_output_ids_path, "rb") as f:
        dumped_ids_list = pkl.load(f)
    for dumped_ids in dumped_ids_list:
        assert len(dumped_ids["ref_output_ids"]) == len(dumped_ids["output_ids_lst"]) == 1

        ref_text = tokenizer.batch_decode(dumped_ids["ref_output_ids"][0], skip_special_tokens=True)
        print(f"reference: {ref_text}")

        out_text = tokenizer.batch_decode(dumped_ids["output_ids_lst"][0], skip_special_tokens=True)
        print(f"Thor output: {out_text}")
        
        print("###############\n")
