import json
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--subtrack",
        type=str,
        default='base',
        choices=['base', 'large']
    )
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    
    subtrack = args.subtrack
    source_path = f'process/{subtrack}_our_testcase_purify.json'
    with open(source_path, 'r') as f:
        source = json.load(f)
    norepe = {}

    for goal, test_cases in source.items():
        norepe[goal] = []
        for test_case in test_cases:
            flag = True
            for x in norepe[goal]:
                if test_case==x:
                    flag = False
            if flag:
                norepe[goal].append(test_case)
                
    dest_path1 = f'process/{subtrack}_our_testcase_sample_5.json'
    dest_path2 = f'{subtrack}_our_testcase.json'

    dest = {}
    for goal, test_cases in norepe.items():
        dest[goal] = test_cases[:50]
        if(len(test_cases) == 0):
            dest[goal] = [goal for i in range(50)]
            continue
        idxs = np.random.randint(0, len(test_cases[:5]), size=50-len(dest[goal]))
        for i in idxs:
            dest[goal].append(test_cases[i])
        
    with open(dest_path1, 'w') as f:
        json.dump(dest, f, indent=4) 
    with open(dest_path2, 'w') as f:
        json.dump(dest, f, indent=4) 
        
if __name__ == "__main__":
    main()