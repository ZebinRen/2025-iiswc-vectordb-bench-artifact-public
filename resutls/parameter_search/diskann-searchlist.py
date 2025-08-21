import argparse
import os

import parse
from proc import *

'''
Run the experiments:
python3 diskann-searchlist.py --case-type cohere-1m --run > diskann-para-search-cohere-1m.log 2>&1 &
python3 diskann-searchlist.py --case-type cohere-10m --run > diskann-para-search-cohere-10m.log 2>&1 &
python3 diskann-searchlist.py --case-type openai-500k --run > diskann-para-search-openai-500k.log 2>&1 &
python3 diskann-searchlist.py --case-type openai-5m --run > diskann-para-search-openai-5m.log 2>&1 &
'''

vectordb_bin = '/home/zebin/anaconda3/envs/vectordb-bench-new/bin/vectordbbench'
searchlist_min = 10
searchlist_max = 100
target_recall = 0.9
results_dir = 'results-diskann-para-search-searchlist'

database_config = {
    'cohere-1m': {
        'uri': 'http://localhost:19539',
        'db-label': 'milvus_diskann_cohere_1m',
        'case-type': 'Performance768D1M',
    },
    'cohere-10m': {
        'uri': 'http://localhost:19540',
        'db-label': 'milvus_diskann_cohere_10m',
        'case-type': 'Performance768D10M',
    },
    'openai-500k': {
        'uri': 'http://localhost:19541',
        'db-label': 'milvus_diskann_openai_500k',
        'case-type': 'Performance1536D500K',
    },
    'openai-5m': {
        'uri': 'http://localhost:19542',
        'db-label': 'milvus_diskann_openai_5m',
        'case-type': 'Performance1536D5M',
    },
}

def get_recall(filepath):
    res = parse.parse_vectordb_bench_output(filepath)
    
    return res['recall']
    

parser = argparse.ArgumentParser()
parser.add_argument('-r',
                    '--run',
                    action='store_true',
                    help='Run the experiment')
parser.add_argument('--case-type', type=str, default='', help='Case type for the benchmark')
args = parser.parse_args()


RUN = args.run
assert args.case_type in ['cohere-1m', 'cohere-10m', 'openai-500k', 'openai-5m']
cur_config = database_config[args.case_type]
results_dir += f'-{cur_config["case-type"]}'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

cmd_common = f'{vectordb_bin} milvusdiskann --skip-drop-old --skip-load --db-label {cur_config['db-label']} --case-type {cur_config['case-type']} --k 10 --num-concurrency 1 --uri {cur_config['uri']} --password "" '

searchlist_min_output = os.path.join(results_dir, f'diskann-para-search-{cur_config["case-type"]}-searchlist-{searchlist_min}.txt')
searchlist_min_cmd = f'{cmd_common} --search-list {searchlist_min} > {searchlist_min_output} 2>&1'
searchlist_max_output = os.path.join(results_dir, f'diskann-para-search-{cur_config["case-type"]}-searchlist-{searchlist_max}.txt')
searchlist_max_cmd = f'{cmd_common} --search-list {searchlist_max} > {searchlist_max_output} 2>&1'

print("[INFO] Starting parameter search for diskann k-probe...")
print("[INFO] min searchlist =", searchlist_min)
print("[INFO] max searchlist =", searchlist_max)
exec_cmd(searchlist_min_cmd, RUN)
exec_cmd(searchlist_max_cmd, RUN)
searchlist_min_recall = get_recall(searchlist_min_output)
searchlist_max_recall = get_recall(searchlist_max_output)

print(f"[RESULT] searchlist = {searchlist_min}, recall = {searchlist_min_recall}")
print(f"[RESULT] searchlist = {searchlist_max}, recall = {searchlist_max_recall}")

if searchlist_min_recall >= target_recall:
    print(f"searchlist {searchlist_min} already meets the target recall of {target_recall}. No further search needed.")
    exit(0)
if searchlist_max_recall < target_recall:
    print(f"searchlist {searchlist_max} does not meet the target recall of {target_recall}. No further search needed.")
    exit(0)

while searchlist_max > searchlist_min:
    cur_searchlist = (searchlist_max + searchlist_min) // 2
    cur_output = os.path.join(results_dir, f'diskann-para-search-{cur_config["case-type"]}-searchlist-{cur_searchlist}.txt')
    cur_cmd = f'{cmd_common} --search-list {cur_searchlist} > {cur_output} 2>&1'
    
    print(f"[INFO] Testing searchlist = {cur_searchlist}")
    exec_cmd(cur_cmd, RUN)
    cur_recall = get_recall(cur_output)
    print(f"[RESULT] searchlist = {cur_searchlist}, recall = {cur_recall}")
    
    if cur_recall >= target_recall:
        searchlist_max = cur_searchlist
        searchlist_max_recall = cur_recall
        
    else:
        searchlist_min = cur_searchlist + 1
        print(f"[INFO] evaluate searchlist_min = cur_searchlist + 1 = {searchlist_min}")
        cur_output = os.path.join(results_dir, f'diskann-para-search-{cur_config["case-type"]}-searchlist-{searchlist_min}.txt')
        cur_cmd = f'{cmd_common} --search-list {searchlist_min} > {cur_output} 2>&1'
        exec_cmd(cur_cmd, RUN)
        searchlist_min_recall = get_recall(cur_output)
        print(f"[RESULT] searchlist_min = {searchlist_min}, recall = {searchlist_min_recall}")
        if searchlist_min_recall >= target_recall:
            break
        
print(f"[INFO] Final searchlist value: searchlist = {searchlist_min}, recall = {searchlist_min_recall}")
print(f"[INFO] Final searchlist value: searchlist = {searchlist_max}, recall = {searchlist_max_recall}")
    
    
    


