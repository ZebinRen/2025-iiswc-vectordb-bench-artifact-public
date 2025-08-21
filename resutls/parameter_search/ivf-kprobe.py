import argparse
import os

import parse
from proc import *

'''
Run the experiments:
nohup python3 ivf-kprobe.py --case-type cohere-1m --run > ivf-para-search-cohere-1m.log 2>&1 &
nohup python3 ivf-kprobe.py --case-type cohere-10m --run > ivf-para-search-cohere-10m.log 2>&1 &
nohup python3 ivf-kprobe.py --case-type openai-500k --run > ivf-para-search-openai-500k.log 2>&1 &
nohup python3 ivf-kprobe.py --case-type openai-5m --run > ivf-para-search-openai-5m.log 2>&1 &
'''

vectordb_bin = '/home/zebin/anaconda3/envs/vectordb-bench-new/bin/vectordbbench'
probe_min = 1
probe_max = 100
target_recall = 0.9
results_dir = 'results-ivf-para-search-kprobe'

database_config = {
    'cohere-1m': {
        'uri': 'http://localhost:19531',
        'db-label': 'milvus_ivf_cohere_1m',
        'case-type': 'Performance768D1M',
        'lists': 4000,
    },
    'cohere-10m': {
        'uri': 'http://localhost:19532',
        'db-label': 'milvus_ivf_cohere_10m',
        'case-type': 'Performance768D10M',
        'lists': 12648,
    },
    'openai-500k': {
        'uri': 'http://localhost:19533',
        'db-label': 'milvus_ivf_openai_500k',
        'case-type': 'Performance1536D500K',
        'lists': 2828,
    },
    'openai-5m': {
        'uri': 'http://localhost:19534',
        'db-label': 'milvus_ivf_openai_5m',
        'case-type': 'Performance1536D5M',
        'lists': 8944,
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

cmd_common = f'{vectordb_bin} milvusivfflat --skip-drop-old --skip-load --db-label {cur_config['db-label']} --case-type {cur_config['case-type']} --k 10 --num-concurrency 1 --uri {cur_config['uri']} --password "" --lists {cur_config['lists']}'

probe_min_output = os.path.join(results_dir, f'ivf-para-search-{cur_config["case-type"]}-probe-{probe_min}.txt')
probe_min_cmd = f'{cmd_common} --probes {probe_min} > {probe_min_output} 2>&1'
probe_max_output = os.path.join(results_dir, f'ivf-para-search-{cur_config["case-type"]}-probe-{probe_max}.txt')
probe_max_cmd = f'{cmd_common} --probes {probe_max} > {probe_max_output} 2>&1'

print("[INFO] Starting parameter search for IVF k-probe...")
print("[INFO] min probe =", probe_min)
print("[INFO] max probe =", probe_max)
exec_cmd(probe_min_cmd, RUN)
exec_cmd(probe_max_cmd, RUN)
probe_min_recall = get_recall(probe_min_output)
probe_max_recall = get_recall(probe_max_output)

print(f"[RESULT] probe = {probe_min}, recall = {probe_min_recall}")
print(f"[RESULT] probe = {probe_max}, recall = {probe_max_recall}")

if probe_min_recall >= target_recall:
    print(f"Probe {probe_min} already meets the target recall of {target_recall}. No further search needed.")
    exit(0)
if probe_max_recall < target_recall:
    print(f"Probe {probe_max} does not meet the target recall of {target_recall}. No further search needed.")
    exit(0)

# cur_probe_cmd = '--probes 30'
while probe_max > probe_min:
    cur_probe = (probe_max + probe_min) // 2
    cur_output = os.path.join(results_dir, f'ivf-para-search-{cur_config["case-type"]}-probe-{cur_probe}.txt')
    cur_cmd = f'{cmd_common} --probes {cur_probe} > {cur_output} 2>&1'
    
    print(f"[INFO] Testing probe = {cur_probe}")
    exec_cmd(cur_cmd, RUN)
    cur_recall = get_recall(cur_output)
    print(f"[RESULT] probe = {cur_probe}, recall = {cur_recall}")
    
    if cur_recall >= target_recall:
        probe_max = cur_probe
        probe_max_recall = cur_recall
        
    else:
        probe_min = cur_probe + 1
        print(f"[INFO] evaluate probe_min = cur_probe + 1 = {probe_min}")
        cur_output = os.path.join(results_dir, f'ivf-para-search-{cur_config["case-type"]}-probe-{probe_min}.txt')
        cur_cmd = f'{cmd_common} --probes {probe_min} > {cur_output} 2>&1'
        exec_cmd(cur_cmd, RUN)
        probe_min_recall = get_recall(cur_output)
        print(f"[RESULT] probe_min = {probe_min}, recall = {probe_min_recall}")
        if probe_min_recall >= target_recall:
            break
        
print(f"[INFO] Final probe value: probe = {probe_min}, recall = {probe_min_recall}")
print(f"[INFO] Final probe value: probe = {probe_max}, recall = {probe_max_recall}")
    
    
    


