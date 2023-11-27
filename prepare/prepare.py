import json
import numpy as np

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--source_path",
        type=str,
        default=None
    )
    parser.add_argument(
        "--output_testcase_path",
        type=str,
        default=None
    )
    
    parser.add_argument(
        "--output_icl_dataset_path",
        type=str,
        default=None
    )
    
    parser.add_argument(
        "--output_prefix_dataset_path",
        type=str,
        default=None
    )
    
    parser.add_argument(
        "--output_surffix_dataset_path",
        type=str,
        default=None
    )
    
    
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    source_path = args.source_path
    output_testcase_path = args.output_testcase_path
    output_icl_dataset_path = args.output_icl_dataset_path
    output_prefix_dataset_path = args.output_prefix_dataset_path
    output_surffix_dataset_path = args.output_surffix_dataset_path
    
    # testcase file initialization
    with open(source_path, 'r') as f:
        source = json.load(f)
    output_testcase = {}
    for goal in source:
        output_testcase[goal] = []
    with open(output_testcase_path, 'w') as f:
        json.dump(output_testcase, f, indent=4)
    
    # prefix dataset file initialization
    output_prefix = []
    for i in range(50):
        prefix = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"
        output_prefix.append(prefix)
    with open(output_prefix_dataset_path, 'w') as f:
        json.dump(output_prefix, f, indent=4)
        
    # surffix dataset file initialization
    output_surffix = ["! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"]
    with open(output_surffix_dataset_path, 'w') as f:
        json.dump(output_surffix, f, indent=4)
        
    # icl dataset file initialization
    output_icl = []
    with open(output_icl_dataset_path, 'w') as f:
        json.dump(output_icl, f, indent=4)

if __name__ == "__main__":
    main()