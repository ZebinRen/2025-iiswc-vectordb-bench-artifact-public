"""
Get the performance of Milvus with diskann, only measure the performance without trace.
"""

import os
import numpy as np
import parse


from utils import create_non_exist_path


result_path_root = 'performance-var-arg'
figure_path_root = 'figures'

# variables
all_bwidth = [1, 2, 3, 4, 5, 6]
datasets = ['cohere-1m', 'cohere-10m', 'openai-500k', 'openai-5m']
all_concurrency = [1, 256]
rep = 5


# We only allow these pre-defined arguments
database_config = {
    'cohere-1m': {
        'uri':
        'http://localhost:19539',
        'db-label':
        'milvus_diskann_cohere_1m',
        'case-type':
        'Performance768D1M',
        'db-path':
        '/mnt/vectordb/nvme0n1/milvus/milvus-diskann-cohere-1m',
    },
    'cohere-10m': {
        'uri':
        'http://localhost:19540',
        'db-label':
        'milvus_diskann_cohere_10m',
        'case-type':
        'Performance768D10M',
        'db-path':
        '/mnt/vectordb/nvme0n1/milvus/milvus-diskann-cohere-10m',
    },
    'openai-500k': {
        'uri':
        'http://localhost:19541',
        'db-label':
        'milvus_diskann_openai_500k',
        'case-type':
        'Performance1536D500K',
        'db-path':
        '/mnt/vectordb/nvme0n1/milvus/milvus-diskann-openai-500k',
    },
    'openai-5m': {
        'uri':
        'http://localhost:19542',
        'db-label':
        'milvus_diskann_openai_5m',
        'case-type':
        'Performance1536D5M',
        'db-path':
        '/mnt/vectordb/nvme0n1/milvus/milvus-diskann-openai-5m',
    },
}


## Parse
# Performance results
results_qps = {}
results_latency = {}
results_latency_con_avg = {}
results_latency_con_p99 = {}
results_recall = {}
results_qps_average = {}
results_latency_average = {}
results_latency_con_avg_average = {}
results_latency_con_p99_average = {}
results_recall_average = {}
results_qps_std = {}
results_latency_std = {}
results_latency_con_avg_std = {}
results_latency_con_p99_std = {}
results_recall_std = {}

results_all = [results_qps, results_latency, results_latency_con_avg, results_latency_con_p99, results_recall]
results_aggregated = [results_qps_average, results_latency_average, results_latency_con_avg_average, results_latency_con_p99_average,
                      results_recall_average, results_qps_std, results_latency_std, results_latency_con_avg_std, results_latency_con_p99_std, results_recall_std]

# Trace results
results_cpu_trace = {}
results_mem_trace = {}
results_page_cache_trace = {}

for cur_dataset in datasets:
    for item in results_all:
        item[cur_dataset] = {}
    for item in results_aggregated:
        item[cur_dataset] = {}
    for cur_concurrency in all_concurrency:
        for item in results_all:
            item[cur_dataset][cur_concurrency] = {}
        for item in results_aggregated:
            item[cur_dataset][cur_concurrency] = []
        expr_results_common = os.path.join(result_path_root, f'milvus-diskann-{cur_dataset}-var-bwidth-concurrency-{cur_concurrency}')
        expr_results_path = os.path.join(
            expr_results_common,
            f'milvus-diskann-{cur_dataset}-var-bwidth-concurrency-{cur_concurrency}-output')
        cpu_trace_path = os.path.join(
            expr_results_common,
            f'milvus-diskann-{cur_dataset}-var-bwidth-concurrency-{cur_concurrency}-cpu-trace'
        )
        mem_trace_path = os.path.join(
            expr_results_common,
            f'milvus-diskann-{cur_dataset}-var-bwidth-concurrency-{cur_concurrency}-mem-trace'
        )
        page_cache_trace_path = os.path.join(
            expr_results_common,
            f'milvus-diskann-{cur_dataset}-var-bwidth-concurrency-{cur_concurrency}-page-cache-trace'
        )

        for cur_bwidth in all_bwidth:
            results_qps[cur_bwidth] = []
            results_latency[cur_bwidth] = []
            results_recall[cur_bwidth] = []
            results_latency_con_avg[cur_bwidth] = []
            results_latency_con_p99[cur_bwidth] = []

            qps_cur_rep = []
            latency_cur_rep = []
            recall_cur_rep = []
            latency_con_avg_cur_rep = []
            latency_con_p99_cur_rep = []

            for cur_rep in range(rep):
                # Trace CPU, memory and page cache
                expr_str = f"{cur_dataset}-concurrency-{cur_concurrency}-k-10-searchlist-100-bwidth-{cur_bwidth}-rep-{cur_rep}.log"
                vdb_output_file = f'vdb-bench-{expr_str}'
                vdb_bench_output_path = os.path.join(expr_results_path, vdb_output_file)

                vectordb_perf = parse.parse_vectordb_bench_output(vdb_bench_output_path)

                qps_cur_rep.append(vectordb_perf['qps'])
                latency_cur_rep.append(vectordb_perf['latency'])
                latency_con_avg_cur_rep.append(vectordb_perf['con_latency_avg'])
                latency_con_p99_cur_rep.append(vectordb_perf['con_latency_p99'])
                recall_cur_rep.append(vectordb_perf['recall'])


            results_qps_average[cur_dataset][cur_concurrency].append(
                sum(qps_cur_rep) / len(qps_cur_rep))
            results_latency_average[cur_dataset][cur_concurrency].append(
                sum(latency_cur_rep) / len(latency_cur_rep))
            results_latency_con_avg_average[cur_dataset][
                cur_concurrency].append(sum(latency_con_avg_cur_rep) / len(latency_con_avg_cur_rep))
            results_latency_con_p99_average[cur_dataset][
                cur_concurrency].append(
                    sum(latency_con_p99_cur_rep) /
                    len(latency_con_p99_cur_rep))
            results_recall_average[cur_dataset][cur_concurrency].append(
                sum(recall_cur_rep) / len(recall_cur_rep))
            results_qps_std[cur_dataset][cur_concurrency].append(
                float(np.std(qps_cur_rep)))
            results_latency_std[cur_dataset][cur_concurrency].append(
                float(np.std(latency_cur_rep)))
            results_latency_con_avg_std[cur_dataset][cur_concurrency].append(
                float(np.std(latency_con_avg_cur_rep)))
            results_latency_con_p99_std[cur_dataset][cur_concurrency].append(
                float(np.std(latency_con_p99_cur_rep)))
            results_recall_std[cur_dataset][cur_concurrency].append(
                float(np.std(recall_cur_rep)))


# Print the results first
print("Results:")
print("QPS: ", results_qps)
print("Latency: ", results_latency)
print("Recall: ", results_recall)
print("Average:")
print(f"QPS Average: {results_qps_average}")
print(f"QPS Std Dev: {results_qps_std}")
print(f"Latency Average: {results_latency_average}")
print(f"Latency Std Dev: {results_latency_std}")
print(f"Recall Average: {results_recall_average}")
print(f"Recall Std Dev: {results_recall_std}")

# Print for statistics, only print two digits after the decimal point
print("\nStatistics:")
for cur_dataset in datasets:
    print(f"\nDataset: {cur_dataset}")
    for cur_concurrency in all_concurrency:
        print(f"  Concurrency: {cur_concurrency}")
        printed_qps = [
            f"{x:.2f}"
            for x in results_qps_average[cur_dataset][cur_concurrency]
        ]
        printed_latency = [
            f"{x:.2f}" for x in results_latency_con_avg_average[cur_dataset]
            [cur_concurrency]
        ]
        printed_recall = [
            f"{x:.2f}"
            for x in results_recall_average[cur_dataset][cur_concurrency]
        ]
        print(f"    QPS Average: {printed_qps}")
        print(f"    Latency Average: {printed_latency}")
        print(f"    Recall Average: {printed_recall}")


# Plot
from plot import *

QPS_MAX = 2

figure_name_prefix = f'milvus-diskann-var-bwidth-'
all_figure_dir = os.path.join(figure_path_root)
create_non_exist_path(all_figure_dir)


PLOT_LATENCY = True
if PLOT_LATENCY:
    fig_save_path = os.path.join(all_figure_dir,
                                 f'fig-13-{figure_name_prefix}-p99-latency.pdf')
    group_list = [
        'cohere-1m-con-1', 'cohere-10m-con-1', 'openai-500k-con-1',
        'openai-5m-con-1'
    ]
    y_values = {
        'cohere-1m-con-1': results_latency_con_p99_average['cohere-1m'][1],
        'cohere-10m-con-1': results_latency_con_p99_average['cohere-10m'][1],
        'openai-500k-con-1': results_latency_con_p99_average['openai-500k'][1],
        'openai-5m-con-1': results_latency_con_p99_average['openai-5m'][1],
    }
    std_dev = {
        'cohere-1m-con-1': results_latency_con_p99_std['cohere-1m'][1],
        'cohere-10m-con-1': results_latency_con_p99_std['cohere-10m'][1],
        'openai-500k-con-1': results_latency_con_p99_std['openai-500k'][1],
        'openai-5m-con-1': results_latency_con_p99_std['openai-5m'][1],
    }

    x_ticks = [str(x) for x in all_bwidth]
    legend_label = {
        'cohere-1m-con-1': 'Cohere 1M',
        'cohere-10m-con-1': 'Cohere 10M',
        'openai-500k-con-1': 'OpenAI 500K',
        'openai-5m-con-1': 'OpenAI 5M',
    }
    title = None
    xlabel = 'Beam width'
    ylabel = 'Latency (ms)'

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
    ax.set_xlim(0, 6.5)
    ax.set_ylim(0, 25)

    for (index, group_name) in zip(range(len(group_list)), group_list):
        # x, y, std_dev, data_label = data[group_name]
        x = range(1, len(y_values[group_name]) + 1)
        y = y_values[group_name]
        yerr = None
        if std_dev:
            yerr = std_dev[group_name]

        plt.errorbar(
            x,
            y,
            yerr=yerr,
            label=legend_label[group_name],
            marker=dot_style[index % len(dot_style)],
            linewidth=linewidth,
            markersize=markersize,
            color=get_next_color(),
        )
        # Add data label
        # for i in range(len(data_label)):
        #     ax.text(x[i], y[i], data_label[i], size=datalabel_size)

    if legend_label != None:
        plt.legend(fontsize=legend_font_size,
                   labelspacing=0.1,
                   ncol=2,
                   columnspacing=0.1,
                   loc='upper left',
                   bbox_to_anchor=(0, 0.3))
        # plt.legend(fontsize=legend_font_size,
        #            ncol=2,
        #            loc='upper left',
        #            bbox_to_anchor=(0, 1.2),
        #            columnspacing=0.3)

    plt.savefig(fig_save_path, bbox_inches='tight')
    plt.close()

PLOT_ACC = True
if PLOT_ACC:
    # Data, set unused value to none
    fig_save_path = os.path.join(all_figure_dir,
                                 f'{figure_name_prefix}-acc.pdf')
    group_list = ['cohere-1m', 'cohere-10m', 'openai-500k', 'openai-5m']
    y_values = {
        'cohere-1m': results_recall_average['cohere-1m'][1],
        'cohere-10m': results_recall_average['cohere-10m'][1],
        'openai-500k': results_recall_average['openai-500k'][1],
        'openai-5m': results_recall_average['openai-5m'][1],
    }
    std_dev = None
    legend_label = {
        'cohere-1m': 'Cohere 1M',
        'cohere-10m': 'Cohere 10M',
        'openai-500k': 'OpenAI 500K',
        'openai-5m': 'OpenAI 5M',
    }

    title = None
    xlabel = 'Beam width'
    ylabel = 'recall@10'

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
    ax.set_xlim(0, 6.5)
    ax.set_ylim(0.9, 1)

    for (index, group_name) in zip(range(len(group_list)), group_list):
        # x, y, std_dev, data_label = data[group_name]
        x = range(1, len(y_values[group_name]) + 1)
        y = y_values[group_name]
        yerr = None
        if std_dev:
            yerr = std_dev[group_name]

        plt.errorbar(
            x,
            y,
            yerr=yerr,
            label=legend_label[group_name],
            marker=dot_style[index % len(dot_style)],
            linewidth=linewidth,
            markersize=markersize,
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


PLOT_QPS_ALL = True
if PLOT_QPS_ALL:
    # Data, set unused value to none
    fig_save_path = os.path.join(all_figure_dir,
                                    f'fig-12-{figure_name_prefix}-cohere-qps-all.pdf')
    group_list = [
        'cohere-1m-con-1',
        'cohere-10m-con-1',
        'openai-500k-con-1',
        'openai-5m-con-1',
        'cohere-1m-con-256',
        'cohere-10m-con-256',
        'openai-500k-con-256',
        'openai-5m-con-256',
    ]
    y_values = {
        'cohere-1m-con-1':
        [x / 1000 for x in results_qps_average['cohere-1m'][1]],
        'cohere-1m-con-256':
        [x / 1000 for x in results_qps_average['cohere-1m'][256]],
        'cohere-10m-con-1':
        [x / 1000 for x in results_qps_average['cohere-10m'][1]],
        'cohere-10m-con-256':
        [x / 1000 for x in results_qps_average['cohere-10m'][256]],
        'openai-500k-con-1':
        [x / 1000 for x in results_qps_average['openai-500k'][1]],
        'openai-500k-con-256':
        [x / 1000 for x in results_qps_average['openai-500k'][256]],
        'openai-5m-con-1':
        [x / 1000 for x in results_qps_average['openai-5m'][1]],
        'openai-5m-con-256':
        [x / 1000 for x in results_qps_average['openai-5m'][256]],

    }
    std_dev = {
        'cohere-1m-con-1': [x / 1000 for x in results_qps_std['cohere-1m'][1]],
        'cohere-1m-con-256':
        [x / 1000 for x in results_qps_std['cohere-1m'][256]],
        'cohere-10m-con-1':
        [x / 1000 for x in results_qps_std['cohere-10m'][1]],
        'cohere-10m-con-256':
        [x / 1000 for x in results_qps_std['cohere-10m'][256]],
        'openai-500k-con-1':
        [x / 1000 for x in results_qps_std['openai-500k'][1]],
        'openai-500k-con-256':
        [x / 1000 for x in results_qps_std['openai-500k'][256]],
        'openai-5m-con-1': [x / 1000 for x in results_qps_std['openai-5m'][1]],
        'openai-5m-con-256':
        [x / 1000 for x in results_qps_std['openai-5m'][256]],
    }
    x_ticks = [str(x) for x in all_bwidth]

    title = None
    xlabel = 'Beam width'
    ylabel = 'QPS (K)'

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

    ax.tick_params(axis='both',
                    which='major',
                    labelsize=axis_tick_font_size)
    ax.xaxis.set_ticks(range(1, len(x_ticks) + 1))
    ax.set_xticklabels(x_ticks)
    ax.set_xlim(0, 6.5)
    ax.set_ylim(0, QPS_MAX)

    for (index, group_name) in zip(range(len(group_list)), group_list):
        # x, y, std_dev, data_label = data[group_name]
        x = range(1, len(y_values[group_name]) + 1)
        y = y_values[group_name]
        yerr = None
        if std_dev:
            yerr = std_dev[group_name]

        if group_name[-1] == '1':
            linestyle = 'solid'
        else:
            linestyle = 'dashed'

        if group_name == 'cohere-1m-con-256':
            reset_color()

        plt.errorbar(
            x,
            y,
            yerr=yerr,
            marker=dot_style[index % len(dot_style)],
            linewidth=linewidth,
            linestyle=linestyle,
            markersize=markersize,
            color=get_next_color(),
        )
        # Add data label
        # for i in range(len(data_label)):
        #     ax.text(x[i], y[i], data_label[i], size=datalabel_size)

    legend_label = {
        'Cohere 1M': 'Cohere 1M',
        'Cohere 10M': 'Cohere 10M',
        'OpenAI 500K': 'OpenAI 500K',
        'OpenAI 5M': 'OpenAI 5M',
    }

    reset_color()
    for cur_label in legend_label:
        plt.errorbar(
            [], [],
            marker=dot_style[index % len(dot_style)],
            label=cur_label,
            linewidth=linewidth,
            linestyle='solid',
            markersize=markersize,
            color=get_next_color(),
        )

    reset_color()
    plt.errorbar(
        [],
        [],
        label='Concurrency 1',
        linewidth=linewidth,
        linestyle='solid',
        markersize=markersize,
        color=get_next_color(),
    )
    reset_color()
    plt.errorbar(
        [],
        [],
        label='Concurrency 256',
        linewidth=linewidth,
        linestyle='dashed',
        markersize=markersize,
        color=get_next_color(),
    )


    if legend_label != None:
        plt.legend(
            fontsize=legend_font_size,
            labelspacing=0.1,
            ncol=2,
            columnspacing=0.1,
            bbox_to_anchor=(0, 0.5),
            loc='upper left',
        )
        # plt.legend(fontsize=legend_font_size,
        #            ncol=2,
        #            loc='upper left',
        #            bbox_to_anchor=(0, 1.2),
        #            columnspacing=0.3)

    plt.savefig(fig_save_path, bbox_inches='tight')
    plt.close()
