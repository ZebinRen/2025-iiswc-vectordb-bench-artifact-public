import argparse
import os

import parse
from proc import *
'''
Run the experiments:
python3 lancedb-hnsw-ef-search.py --case-type cohere-1m --run > lancedb-hnsw-para-search-cohere-1m.log 2>&1 &
python3 lancedb-hnsw-ef-search.py --case-type cohere-10m --run > lancedb-hnsw-para-search-cohere-10m.log 2>&1 &
python3 lancedb-hnsw-ef-search.py --case-type openai-500k --run > lancedb-hnsw-para-search-openai-500k.log 2>&1 &
python3 lancedb-hnsw-ef-search.py --case-type openai-5m --run > lancedb-hnsw-para-search-openai-5m.log 2>&1 &
'''

bench_bin = '/home/zebin/anaconda3/envs/vectordb-bench-new/bin/vectordbbench lancedbhnsw --skip-drop-old --skip-load'
efsearch_min = 10
efsearch_max = 100
target_recall = 0.9
results_dir = 'results-lancedb-hnsw-para-search-efsearch'

database_config = {
    'cohere-1m': {
        'uri': '/mnt/vectordb/nvme0n1/lancedb/lancedb-hnsw-cohere-1m',
        'case-type': 'Performance768D1M',
    },
    'cohere-10m': {
        'uri': '/mnt/vectordb/nvme0n1/lancedb/lancedb-hnsw-cohere-10m',
        'case-type': 'Performance768D10M',
    },
    'openai-500k': {
        'uri': '/mnt/vectordb/nvme0n1/lancedb/lancedb-hnsw-openai-500k',
        'case-type': 'Performance1536D500K',
    },
    'openai-5m': {
        'uri': '/mnt/vectordb/nvme0n1/lancedb/lancedb-hnsw-openai-5m',
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
parser.add_argument('--case-type',
                    type=str,
                    default='',
                    help='Case type for the benchmark')
args = parser.parse_args()

RUN = args.run
assert args.case_type in [
    'cohere-1m', 'cohere-10m', 'openai-500k', 'openai-5m'
]
cur_config = database_config[args.case_type]
results_dir += f'-{cur_config["case-type"]}'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

cmd_common = f"{bench_bin} --case-type {cur_config['case-type']} --k 10 --num-concurrency 1 --uri {cur_config['uri']} --m 16 --ef-construction 200"


efsearch_min_output = os.path.join(
    results_dir,
    f'hnsw-para-search-{cur_config["case-type"]}-efsearch-{efsearch_min}.txt')
efsearch_min_cmd = f'{cmd_common} --ef {efsearch_min} > {efsearch_min_output} 2>&1'
efsearch_max_output = os.path.join(
    results_dir,
    f'hnsw-para-search-{cur_config["case-type"]}-efsearch-{efsearch_max}.txt')
efsearch_max_cmd = f'{cmd_common} --ef {efsearch_max} > {efsearch_max_output} 2>&1'

print("[INFO] Starting parameter search for HNSW efsearch...")
print("[INFO] min efsearch =", efsearch_min)
print("[INFO] max efsearch =", efsearch_max)
exec_cmd(efsearch_min_cmd, RUN)
exec_cmd(efsearch_max_cmd, RUN)
efsearch_min_recall = get_recall(efsearch_min_output)
efsearch_max_recall = get_recall(efsearch_max_output)

print(f"[RESULT] efsearch = {efsearch_min}, recall = {efsearch_min_recall}")
print(f"[RESULT] efsearch = {efsearch_max}, recall = {efsearch_max_recall}")

if efsearch_min_recall >= target_recall:
    print(
        f"efsearch {efsearch_min} already meets the target recall of {target_recall}. No further search needed."
    )
    exit(0)
if efsearch_max_recall < target_recall:
    print(
        f"efsearch {efsearch_max} does not meet the target recall of {target_recall}. No further search needed."
    )
    exit(0)

while efsearch_max > efsearch_min:
    cur_efsearch = (efsearch_max + efsearch_min) // 2
    cur_output = os.path.join(
        results_dir,
        f'hnsw-para-search-{cur_config["case-type"]}-efsearch-{cur_efsearch}.txt'
    )
    cur_cmd = f'{cmd_common} --ef {cur_efsearch} > {cur_output} 2>&1'

    print(f"[INFO] Testing efsearch = {cur_efsearch}")
    exec_cmd(cur_cmd, RUN)
    cur_recall = get_recall(cur_output)
    print(f"[RESULT] efsearch = {cur_efsearch}, recall = {cur_recall}")

    if cur_recall >= target_recall:
        efsearch_max = cur_efsearch
        efsearch_max_recall = cur_recall

    else:
        efsearch_min = cur_efsearch + 1
        print(
            f"[INFO] evaluate efsearch_min = cur_efsearch + 1 = {efsearch_min}"
        )
        cur_output = os.path.join(
            results_dir,
            f'hnsw-para-search-{cur_config["case-type"]}-efsearch-{efsearch_min}.txt'
        )
        cur_cmd = f'{cmd_common} --ef {efsearch_min} > {cur_output} 2>&1'
        exec_cmd(cur_cmd, RUN)
        efsearch_min_recall = get_recall(cur_output)
        print(
            f"[RESULT] efsearch_min = {efsearch_min}, recall = {efsearch_min_recall}"
        )
        if efsearch_min_recall >= target_recall:
            break

print(
    f"[INFO] Final efsearch value: efsearch = {efsearch_min}, recall = {efsearch_min_recall}"
)
print(
    f"[INFO] Final efsearch value: efsearch = {efsearch_max}, recall = {efsearch_max_recall}"
)
