import os.path
import numpy as np

import parse
'''
We plot a separate figure for each datasets, both Throughput and Latency.
The current lines that we are drawing: 4 datasets + ?? setups

Datasets:
- cohere-1m
- cohere-10m
- openai-500k
- openai-5m

Database setups:
- milvus-ivfflat
- milvus-hnsw
- milvus-diskann
- lancedbivfpq
- lancedbhnsw
'''

vector_dbs = ['milvus', 'weaviate', 'qdrant', 'lancedb-embedded']
vdb_configs = {
    'milvus': ['ivf', 'hnsw', 'diskann'],
    'qdrant': ['hnsw-mem'],
    'weaviate': ['hnsw'],
    'lancedb-embedded': ['ivf', 'hnsw']
}
vdb_disk = ['milvus-diskann', 'lancedb-embedded-ivf']
datasets = ['cohere-1m', 'cohere-10m', 'openai-500k', 'openai-5m']
all_concurrency = [1, 2, 4, 8, 16, 32, 64, 128, 256]
rep = 5

expr_config_fname = {
    'milvus': {
        'ivf-cohere-1m': 'k-10-nlist-4000-kprobe-25',
        'ivf-cohere-10m': 'k-10-nlist-12648-kprobe-17',
        'ivf-openai-500k': 'k-10-nlist-2828-kprobe-16',
        'ivf-openai-5m': 'k-10-nlist-8944-kprobe-11',
        'hnsw-cohere-1m': 'k-10-m-16-efconstruct-200-efsearch-27',
        'hnsw-cohere-10m': 'k-10-m-16-efconstruct-200-efsearch-43',
        'hnsw-openai-500k': 'k-10-m-16-efconstruct-200-efsearch-14',
        'hnsw-openai-5m': 'k-10-m-16-efconstruct-200-efsearch-10',
        'diskann-cohere-1m': 'k-10-searchlist-10',
        'diskann-cohere-10m': 'k-10-searchlist-10',
        'diskann-openai-500k': 'k-10-searchlist-10',
        'diskann-openai-5m': 'k-10-searchlist-10',
    },
    'qdrant': {
        'hnsw-mem-cohere-1m': 'k-10-m-16-efconstruct-200-efsearch-27',
        'hnsw-mem-cohere-10m': 'k-10-m-16-efconstruct-200-efsearch-43',
        'hnsw-mem-openai-500k': 'k-10-m-16-efconstruct-200-efsearch-14',
        'hnsw-mem-openai-5m': 'k-10-m-16-efconstruct-200-efsearch-10',
    },
    'weaviate': {
        'hnsw-cohere-1m': 'k-10-m-16-efconstruct-200-ef-27',
        'hnsw-cohere-10m': 'k-10-m-16-efconstruct-200-ef-43',
        'hnsw-openai-500k': 'k-10-m-16-efconstruct-200-ef-14',
        'hnsw-openai-5m': 'k-10-m-16-efconstruct-200-ef-10',
    },
    'lancedb-embedded': {
        'ivf-cohere-1m': 'k-10-nlist-4000-kprobe-25',
        'ivf-cohere-10m': 'k-10-nlist-12648-kprobe-17',
        'ivf-openai-500k': 'k-10-nlist-2828-kprobe-16',
        'ivf-openai-5m': 'k-10-nlist-8944-kprobe-11',
        'hnsw-cohere-1m': 'k-10-m-16-efconstruct-200-ef-41',
        'hnsw-cohere-10m': 'k-10-m-16-efconstruct-200-ef-56',
        'hnsw-openai-500k': 'k-10-m-16-efconstruct-200-ef-34',
        'hnsw-openai-5m': 'k-10-m-16-efconstruct-200-ef-38',
    }
}


vector_db_output_dir = 'traces'
figure_dir = 'figures'
os.makedirs(figure_dir, exist_ok=True)

# Parse results
results_cpu_util = {}
results_cpu_util_std = {}

for cur_dataset in datasets:
    results_cpu_util[cur_dataset] = {}
    results_cpu_util_std[cur_dataset] = {}
    for cur_vdb in vector_dbs:
        for cur_index in vdb_configs[cur_vdb]:
            cur_db_index = f"{cur_vdb}-{cur_index}"
            results_cpu_util[cur_dataset][cur_db_index] = []
            results_cpu_util_std[cur_dataset][cur_db_index] = []
            for cur_concurrency in all_concurrency:
                cur_rep_cpu_util = []
                for cur_rep in range(rep):
                    output_file_common = f'{cur_dataset}-con-{cur_concurrency}-{expr_config_fname[cur_vdb][cur_index+"-"+cur_dataset]}-rep-{cur_rep}.log'
                    vdb_output_file = f'vdb-bench-{output_file_common}'
                    cpu_trace_path = os.path.join(
                        vector_db_output_dir,
                        f'{cur_vdb}-{cur_index}-{cur_dataset}',
                        f'{cur_vdb}-{cur_index}-{cur_dataset}-cpu-trace')
                    cpu_trace_output_file = os.path.join(cpu_trace_path, f'cpu-trace-{output_file_common}')
                    # mem_trace_output_file = os.path.join(mem_trace_path, 'mem-trace-' + expr_str)
                    # page_cache_output_file = os.path.join(page_cache_trace_path, 'page-caceh-trace-' + expr_str)
                    vdb_bench_output_path = os.path.join(
                        vector_db_output_dir,
                        f'{cur_vdb}-{cur_index}-{cur_dataset}',
                        f'{cur_vdb}-{cur_index}-{cur_dataset}-output',
                        vdb_output_file)

                    # Check if the file exists before parsing
                    if not os.path.exists(vdb_bench_output_path):
                        # print('File not found:', vdb_bench_output_path)
                        continue

                    # print(vdb_bench_output_path)
                    # print(cpu_trace_output_file)
                    start, end = parse.parse_experiment_times(
                        vdb_bench_output_path)
                    cpu_traces = parse.parse_cpu_trace(cpu_trace_output_file)
                    cpu_idle_during_expr = parse.clip_cpu_trace(
                        cpu_traces, start, end)['idle']
                    cpu_util_during_expr = [100 - float(x) for x in cpu_idle_during_expr]
                    cpu_idle_during_expr = cpu_util_during_expr[10: 20]
                    cpu_util_avg = np.mean(cpu_util_during_expr)
                    cur_rep_cpu_util.append(cpu_util_avg)

                if len(cur_rep_cpu_util) == 0:
                    continue
                results_cpu_util[cur_dataset][cur_db_index].append(np.mean(cur_rep_cpu_util))
                results_cpu_util_std[cur_dataset][cur_db_index].append(np.std(cur_rep_cpu_util))

legend_label = {
    'milvus-ivf': 'Milvus IVF',
    'milvus-hnsw': 'Milvus HNSW',
    'milvus-diskann': 'Milvus DiskANN',
    'qdrant-hnsw-mem': 'Qdrant HNSW',
    'weaviate-hnsw': 'Weaviate HNSW',
    'lancedb-embedded-ivf': 'LanceDB IVFPQ',
    'lancedb-embedded-hnsw': 'LanceDB HNSW',
}

from plot import *

DRAW_COHERE_1M = True

DRAW_COHERE_10M = True
if DRAW_COHERE_10M:
    # QPS
    fig_save_path = os.path.join(figure_dir, 'fig-4-a-cohere-10m-cpu.pdf')
    group_list = []
    for cur_db in vdb_configs:
        for index in vdb_configs[cur_db]:
            group_list.append(f"{cur_db}-{index}")
    y_values = results_cpu_util['cohere-10m']
    std_dev = results_cpu_util_std['cohere-10m']
    x_ticks = [str(x) for x in all_concurrency]
    # legend_label = {i: i for i in group_list}
    print(f"CPU usage cohere 10M: {y_values}")

    title = None
    xlabel = 'Concurrency'
    ylabel = 'CPU Util. (%)'

    # Parameters
    linewidth = 4
    markersize = 15

    datalabel_size = 26
    datalabel_va = 'bottom'
    axis_tick_font_size = 34
    axis_label_font_size = 44
    legend_font_size = 30

    reset_color()
    fig, ax = plt.subplots(figsize=(12, 8))

    plt.xlabel(xlabel, fontsize=axis_label_font_size)
    plt.ylabel(ylabel, fontsize=axis_label_font_size)
    plt.grid(True)

    ax.tick_params(axis='both', which='major', labelsize=axis_tick_font_size)
    ax.xaxis.set_ticks(range(1, len(x_ticks) + 1))
    ax.set_xticklabels(x_ticks)
    # ax.set_xlim()
    ax.set_ylim(0, 100)

    for (index, group_name) in zip(range(len(group_list)), group_list):
        # x, y, std_dev, data_label = data[group_name]
        x = range(1, len(y_values[group_name]) + 1)
        y = y_values[group_name]
        yerr = None
        if std_dev:
            yerr = std_dev[group_name]
        linestype = 'dashed' if group_name in vdb_disk else 'solid'

        plt.errorbar(
            x,
            y,
            yerr=yerr,
            label=legend_label[group_name],
            marker=dot_style[index % len(dot_style)],
            linewidth=linewidth,
            markersize=markersize,
            linestyle=linestype,
            color=get_next_color(),
        )
        # Add data label
        # for i in range(len(data_label)):
        #     ax.text(x[i], y[i], data_label[i], size=datalabel_size)

    if legend_label != None:
        plt.legend(fontsize=legend_font_size, labelspacing=0.1)
        # plt.legend(fontsize=legend_font_size,
        #            ncol=2,
        #            loc='upper left',
        #            bbox_to_anchor=(0, 1.2),
        #            columnspacing=0.3)

    plt.savefig(fig_save_path, bbox_inches='tight')
    plt.close()


DRAW_OPENAI_5M = True
if DRAW_OPENAI_5M:
    # QPS
    fig_save_path = os.path.join(figure_dir, 'fig-4-b-openai-5m-cpu.pdf')
    group_list = []
    for cur_db in vdb_configs:
        for index in vdb_configs[cur_db]:
            group_list.append(f"{cur_db}-{index}")
    y_values = results_cpu_util['openai-5m']
    std_dev = results_cpu_util_std['openai-5m']
    x_ticks = [str(x) for x in all_concurrency]
    # legend_label = {i: i for i in group_list}
    print(f"CPU usage cohere 1M: {y_values}")

    title = None
    xlabel = 'Concurrency'
    ylabel = 'CPU Util. (%)'

    # Parameters
    linewidth = 4
    markersize = 15

    datalabel_size = 26
    datalabel_va = 'bottom'
    axis_tick_font_size = 34
    axis_label_font_size = 44
    legend_font_size = 30

    reset_color()
    fig, ax = plt.subplots(figsize=(12, 8))

    plt.xlabel(xlabel, fontsize=axis_label_font_size)
    plt.ylabel(ylabel, fontsize=axis_label_font_size)
    plt.grid(True)

    ax.tick_params(axis='both', which='major', labelsize=axis_tick_font_size)
    ax.xaxis.set_ticks(range(1, len(x_ticks) + 1))
    ax.set_xticklabels(x_ticks)
    # ax.set_xlim()
    ax.set_ylim(0, 100)

    for (index, group_name) in zip(range(len(group_list)), group_list):
        # x, y, std_dev, data_label = data[group_name]
        x = range(1, len(y_values[group_name]) + 1)
        y = y_values[group_name]
        yerr = None
        if std_dev:
            yerr = std_dev[group_name]
        linestype = 'dashed' if group_name in vdb_disk else 'solid'

        plt.errorbar(
            x,
            y,
            yerr=yerr,
            label=legend_label[group_name],
            marker=dot_style[index % len(dot_style)],
            linewidth=linewidth,
            markersize=markersize,
            linestyle=linestype,
            color=get_next_color(),
        )
        # Add data label
        # for i in range(len(data_label)):
        #     ax.text(x[i], y[i], data_label[i], size=datalabel_size)

    # if legend_label != None:
    #     plt.legend(fontsize=legend_font_size, labelspacing=0.1)
    #     # plt.legend(fontsize=legend_font_size,
    #     #            ncol=2,
    #     #            loc='upper left',
    #     #            bbox_to_anchor=(0, 1.2),
    #     #            columnspacing=0.3)

    plt.savefig(fig_save_path, bbox_inches='tight')
    plt.close()
