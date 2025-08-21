"""
Get the performance of Milvus with diskann, only measure the performance without trace.
"""
import argparse
from datetime import datetime
import os
import time
import signal
import subprocess
import numpy as np
import yaml
import pickle

import parse
import proc

from utils import create_non_exist_path


DEBUG = False
result_path_root = 'results-io-trace-var-arg'
figure_path_root = 'figures'
parsed_file_dataset = 'milvus-diskann-var-bwidth-io-trace-parsed.bin'

# variables
all_bwidth = [1, 2, 3, 4, 5, 6]
rep = 5
all_datasets = ["cohere-1m", "cohere-10m", "openai-500k", "openai-5m"]
all_concurrency = [1, 256]

# Debug set
if DEBUG:
    all_bwidth = [1, 2]
    rep = 2

# We only allow these pre-defined arguments
database_config = {
    'cohere-1m': {
        'uri': 'http://localhost:19539',
        'db-label': 'milvus_diskann_cohere_1m',
        'case-type': 'Performance768D1M',
        'db-path': '/mnt/vectordb/nvme0n1/milvus/milvus-diskann-cohere-1m',
    },
    'cohere-10m': {
        'uri': 'http://localhost:19540',
        'db-label': 'milvus_diskann_cohere_10m',
        'case-type': 'Performance768D10M',
        'db-path': '/mnt/vectordb/nvme0n1/milvus/milvus-diskann-cohere-10m',
    },
    'openai-500k': {
        'uri': 'http://localhost:19541',
        'db-label': 'milvus_diskann_openai_500k',
        'case-type': 'Performance1536D500K',
        'db-path': '/mnt/vectordb/nvme0n1/milvus/milvus-diskann-openai-500k',
    },
    'openai-5m': {
        'uri': 'http://localhost:19542',
        'db-label': 'milvus_diskann_openai_5m',
        'case-type': 'Performance1536D5M',
        'db-path': '/mnt/vectordb/nvme0n1/milvus/milvus-diskann-openai-5m',
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

# Trace results
results_start_end_offset = {}
results_bandwidth_trace = {}

results_total_bandwidth_avg = {}
results_average_bandwidth_avg = {}

results_total_bandwidth_std = {}
results_average_bandwidth_std = {}

all_results_plain = [
    results_qps, results_latency, results_latency_con_avg,
    results_latency_con_p99, results_recall
]
all_results_aggregated = [
    results_qps_average, results_latency_average,
    results_latency_con_avg_average, results_latency_con_p99_average,
    results_recall_average, results_qps_std, results_latency_std,
    results_latency_con_avg_std, results_latency_con_p99_std,
    results_recall_std, results_total_bandwidth_avg,
    results_average_bandwidth_avg, results_total_bandwidth_std,
    results_average_bandwidth_std
]

if os.path.exists(parsed_file_dataset):
    f = open(parsed_file_dataset, 'rb')
    for item in all_results_plain:
        item.update(pickle.load(f))
    for item in all_results_aggregated:
        item.update(pickle.load(f))
    f.close()

else:
    for cur_dataset in all_datasets:
        for item in all_results_plain:
            item[cur_dataset] = {}
        for item in all_results_aggregated:
            item[cur_dataset] = {}
        for concurrency in all_concurrency:
            expr_results_common = os.path.join(
                result_path_root,
                f'milvus-diskann-{cur_dataset}-var-bwidth-concurrency-{concurrency}-io-trace'
            )
            expr_results_path = os.path.join(
                expr_results_common,
                f'milvus-diskann-{cur_dataset}-var-bwidth-concurrency-{concurrency}-output')
            bio_trace_path = os.path.join(
                expr_results_common,
                f'milvus-diskann-{cur_dataset}-var-bwidth-concurrency-{concurrency}-io-trace'
            )

            for item in all_results_plain:
                item[cur_dataset][concurrency] = {}
            for item in all_results_aggregated:
                item[cur_dataset][concurrency] = []


            for cur_bwidth in all_bwidth:
                for item in all_results_plain:
                    item[cur_dataset][concurrency][cur_bwidth] = []
                cur_rep_total_bandwidth = []
                cur_rep_average_bandwidth = []
                for cur_rep in range(rep):
                    expr_str = f"{cur_dataset}-concurrency-{concurrency}-k-10-searchlist-100-bwidth-{cur_bwidth}-rep-{cur_rep}.log"
                    vdb_output_file = f'vdb-bench-{expr_str}'
                    vdb_bench_output_path = os.path.join(
                        expr_results_path, vdb_output_file)
                    bio_trace_output_path = os.path.join(
                        bio_trace_path, 'io-trace-' + expr_str)

                    # print for debug
                    print(
                        f"Parsing: beamwidth {cur_bwidth}, rep {cur_rep}")

                    # Parse the performance
                    vectordb_perf = parse.parse_vectordb_bench_output(
                        vdb_bench_output_path)

                    results_qps[cur_dataset][concurrency][
                        cur_bwidth].append(vectordb_perf['qps'])
                    results_latency[cur_dataset][concurrency][
                        cur_bwidth].append(vectordb_perf['latency'])
                    results_latency_con_avg[cur_dataset][concurrency][
                        cur_bwidth].append(
                            vectordb_perf['con_latency_avg'])
                    results_latency_con_p99[cur_dataset][concurrency][
                        cur_bwidth].append(
                            vectordb_perf['con_latency_p99'])
                    results_recall[cur_dataset][concurrency][
                        cur_bwidth].append(vectordb_perf['recall'])

                    # Parse the trace
                    vdb_start, vdb_end = parse.parse_experiment_times(
                        vdb_bench_output_path)
                    trace_file = open(bio_trace_output_path, 'r')
                    line = trace_file.readline().strip()
                    line = trace_file.readline().strip()  # skip the first line
                    trace_start = line.split()[-1]
                    fmt = "%H:%M:%S"
                    vdb_start_offset_in_trace = (
                        datetime.strptime(vdb_start, fmt) -
                        datetime.strptime(trace_start, fmt)).total_seconds()
                    vdb_end_offset_in_trace = (
                        datetime.strptime(vdb_end, fmt) -
                        datetime.strptime(trace_start, fmt)).total_seconds()

                    timestamps, op, bite_size, start_sector, num_sectors = parse.parse_bite_size(
                        bio_trace_output_path
                    )  # timestamps is already in seconds
                    bite_size_by_op, bite_size_aggregated, io_size_by_sec = parse.process_bite_size(
                        timestamps, op, bite_size)
                    cur_read_bandwidth_trace = [
                        x / 1024 for x in io_size_by_sec['ALL_READS']['size']
                    ]
                    if rep == 0:
                        # for the traces, we only add rep = 0
                        results_start_end_offset[cur_dataset][concurrency][
                            cur_bwidth] = (vdb_start_offset_in_trace,
                                               vdb_end_offset_in_trace)
                        results_bandwidth_trace[cur_dataset][concurrency][
                            cur_bwidth] = cur_read_bandwidth_trace

                    # print(f"vdb_start: {vdb_start}, vdb_end: {vdb_end}, trace_start: {trace_start}")
                    # print(f"vdb_start_offset_in_trace: {vdb_start_offset_in_trace}, vdb_end_offset_in_trace: {vdb_end_offset_in_trace}")
                    # print(f"cur_read_bandwidth_trace length: {len(cur_read_bandwidth_trace)}")
                    # print(f"cur_bandwidth trace: {cur_read_bandwidth_trace}")
                    # get the average and per-qps bandwidth
                    clipped_bandwidth = cur_read_bandwidth_trace[
                        int(vdb_start_offset_in_trace
                            ):int(vdb_end_offset_in_trace) + 1]
                    # bandwidth results
                    cur_total_bandwidth = sum(clipped_bandwidth) / len(
                        clipped_bandwidth)
                    cur_average_bandwidth = cur_total_bandwidth / vectordb_perf[
                        'qps']
                    cur_rep_total_bandwidth.append(cur_total_bandwidth)
                    cur_rep_average_bandwidth.append(cur_average_bandwidth)

                # performance
                results_qps_average[cur_dataset][concurrency].append(
                    sum(results_qps[cur_dataset][concurrency][cur_bwidth])
                    /
                    len(results_qps[cur_dataset][concurrency][cur_bwidth]))
                results_latency_average[cur_dataset][concurrency].append(
                    sum(results_latency[cur_dataset][concurrency]
                        [cur_bwidth]) / len(results_latency[cur_dataset]
                                                [concurrency][cur_bwidth]))
                results_latency_con_avg_average[cur_dataset][
                    concurrency].append(
                        sum(results_latency_con_avg[cur_dataset][concurrency]
                            [cur_bwidth]) /
                        len(results_latency_con_avg[cur_dataset][concurrency]
                            [cur_bwidth]))
                results_latency_con_p99_average[cur_dataset][
                    concurrency].append(
                        sum(results_latency_con_p99[cur_dataset][concurrency]
                            [cur_bwidth]) /
                        len(results_latency_con_p99[cur_dataset][concurrency]
                            [cur_bwidth]))
                results_recall_average[cur_dataset][concurrency].append(
                    sum(results_recall[cur_dataset][concurrency]
                        [cur_bwidth]) / len(results_recall[cur_dataset]
                                                [concurrency][cur_bwidth]))
                results_qps_std[cur_dataset][concurrency].append(
                    float(
                        np.std(results_qps[cur_dataset][concurrency]
                               [cur_bwidth])))
                results_latency_std[cur_dataset][concurrency].append(
                    float(
                        np.std(results_latency[cur_dataset][concurrency]
                               [cur_bwidth])))
                results_latency_con_avg_std[cur_dataset][concurrency].append(
                    float(
                        np.std(results_latency_con_avg[cur_dataset]
                               [concurrency][cur_bwidth])))
                results_latency_con_p99_std[cur_dataset][concurrency].append(
                    float(
                        np.std(results_latency_con_p99[cur_dataset]
                               [concurrency][cur_bwidth])))
                results_recall_std[cur_dataset][concurrency].append(
                    float(
                        np.std(results_recall[cur_dataset][concurrency]
                               [cur_bwidth])))

                # bandwidth
                results_total_bandwidth_avg[cur_dataset][concurrency].append(
                    sum(cur_rep_total_bandwidth) /
                    len(cur_rep_total_bandwidth))
                results_average_bandwidth_avg[cur_dataset][concurrency].append(
                    sum(cur_rep_average_bandwidth) /
                    len(cur_rep_average_bandwidth))
                results_total_bandwidth_std[cur_dataset][concurrency].append(
                    float(np.std(cur_rep_total_bandwidth)))
                results_average_bandwidth_std[cur_dataset][concurrency].append(
                    float(np.std(cur_rep_average_bandwidth)))

    f = open(parsed_file_dataset, 'wb')
    for item in all_results_plain:
        pickle.dump(item, f)
    for item in all_results_aggregated:
        pickle.dump(item, f)
    f.close()

# Print with argument values for clarity
print("\nDetailed Results:")
for i, val in enumerate(all_datasets):
    print(f"Dataset: {val}")
    for concurrency in all_concurrency:
        print(f"  Concurrency: {concurrency}")
        for cur_bwidth in all_bwidth:
            qps = results_qps[val][concurrency][cur_bwidth]
            latency = results_latency[val][concurrency][cur_bwidth]
            recall = results_recall[val][concurrency][cur_bwidth]
            print(
                f"    Beamwidth: {cur_bwidth}, QPS: {qps}, Latency: {latency}, Recall: {recall}"
            )

print("Results total bandwidth:")
for cur_dataset in all_datasets:
    print(f"Dataset: {cur_dataset}")
    for concurrency in all_concurrency:
        print(f"  Concurrency: {concurrency}")
        str_list = [
            f'{x:.2f}'
            for x in results_total_bandwidth_avg[cur_dataset][concurrency]
        ]
        std_list = [
            f'{x:.2f}'
            for x in results_total_bandwidth_std[cur_dataset][concurrency]
        ]
        print(f"      {str_list}")
        print(f"      std: {std_list}")

print("Results average bandwidth:")
for cur_dataset in all_datasets:
    print(f"Dataset: {cur_dataset}")
    for concurrency in all_concurrency:
        print(f"  Concurrency: {concurrency}")
        str_list = [
            f'{x:.2f}'
            for x in results_average_bandwidth_avg[cur_dataset][concurrency]
        ]
        std_dev_list = [
            f'{x:.2f}'
            for x in results_average_bandwidth_std[cur_dataset][concurrency]
        ]
        print(f"      {str_list}")
        print(f"      std: {std_dev_list}")


# Plot
from plot import *

PLOT_QPS = True
PLOT_LATENCY = True
PLOT_RECALL = True

TOTAL_BANDWIDTH_MAX_Y = 2000
AVG_BANDWIDTH_MAX_Y = 15

figure_name_prefix = f'milvus-diskann-var-bwidth-io-trace'
all_figure_dir = os.path.join(figure_path_root)
create_non_exist_path(all_figure_dir)

# Total bandwidth: Cohere + OPENAI
PLOT_TOTAL_BANDWIDTH_COHERE = True
if PLOT_TOTAL_BANDWIDTH_COHERE:
    # Data, set unused value to none
    fig_save_path = os.path.join(
        all_figure_dir, f'fig-14-{figure_name_prefix}-total-bandwidth-all.pdf')
    group_list = [
        'cohere-1m-concurrency-1', 'cohere-10m-concurrency-1',
        'openai-500k-concurrency-1', 'openai-5m-concurrency-1',
        'cohere-1m-concurrency-256', 'cohere-10m-concurrency-256',
        'openai-500k-concurrency-256', 'openai-5m-concurrency-256'
    ]
    y_values = {
        'cohere-1m-concurrency-1':
        results_total_bandwidth_avg['cohere-1m'][1],
        'cohere-1m-concurrency-256':
        results_total_bandwidth_avg['cohere-1m'][256],
        'cohere-10m-concurrency-1':
        results_total_bandwidth_avg['cohere-10m'][1],
        'cohere-10m-concurrency-256':
        results_total_bandwidth_avg['cohere-10m'][256],
        'openai-500k-concurrency-1':
        results_total_bandwidth_avg['openai-500k'][1],
        'openai-500k-concurrency-256':
        results_total_bandwidth_avg['openai-500k'][256],
        'openai-5m-concurrency-1':
        results_total_bandwidth_avg['openai-5m'][1],
        'openai-5m-concurrency-256':
        results_total_bandwidth_avg['openai-5m'][256],
    }
    std_dev = {
        'cohere-1m-concurrency-1':
        results_total_bandwidth_std['cohere-1m'][1],
        'cohere-1m-concurrency-256':
        results_total_bandwidth_std['cohere-1m'][256],
        'cohere-10m-concurrency-1':
        results_total_bandwidth_std['cohere-10m'][1],
        'cohere-10m-concurrency-256':
        results_total_bandwidth_std['cohere-10m'][256],
        'openai-500k-concurrency-1':
        results_total_bandwidth_std['openai-500k'][1],
        'openai-500k-concurrency-256':
        results_total_bandwidth_std['openai-500k'][256],
        'openai-5m-concurrency-1':
        results_total_bandwidth_std['openai-5m'][1],
        'openai-5m-concurrency-256':
        results_total_bandwidth_std['openai-5m'][256],
    }
    x_ticks = [str(x) for x in all_bwidth]


    title = None
    xlabel = 'Beam width'
    ylabel = 'Bandwidth (MiB/s)'

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
    ax.set_ylim(0, TOTAL_BANDWIDTH_MAX_Y)
    # Set y-axis ticks at 0, 500, 1000, 1500, 2000
    y_ticks = [0, 500, 1000, 1500, 2000]
    ax.set_yticks(y_ticks)
    y_ticks_label = ['0', '500', '1,000', '1,500', '2,000']
    ax.set_yticklabels(y_ticks_label)

    for (index, group_name) in zip(range(len(group_list)), group_list):
        # x, y, std_dev, data_label = data[group_name]
        x = range(1, len(y_values[group_name]) + 1)
        y = y_values[group_name]
        yerr = None
        if std_dev:
            yerr = std_dev[group_name]


        linestyle = 'solid'
        if group_name.endswith('256'):
            linestyle = 'dashed'
        if group_name == 'cohere-1m-concurrency-256':
            reset_color()
        plt.errorbar(
            x,
            y,
            yerr=yerr,
            # label=legend_label[group_name],
            marker=dot_style[index % len(dot_style)],
            linewidth=linewidth,
            linestyle=linestyle,
            markersize=markersize,
            color=get_next_color(),
        )
        # Add data label
        # for i in range(len(data_label)):
        #     ax.text(x[i], y[i], data_label[i], size=datalabel_size)

    legend_group = [
        'Cohere 1M', 'Cohere 10M', 'OpenAI 500K', 'OpenAI 5M',
    ]

    reset_color()
    for cur_group in legend_group:
        plt.plot(
            [],
            [],
            marker=dot_style[index % len(dot_style)],
            linewidth=linewidth,
            markersize=markersize,
            color=get_next_color(),
            label=cur_group
        )
    for cur_group in ['Concurrecy 1', 'Concurrency 256']:
        reset_color()
        plt.plot(
            [],
            [],
            linewidth=linewidth,
            linestyle='dashed' if cur_group == 'Concurrency 256' else 'solid',
            color=get_next_color(),
            label=cur_group
        )
    if True:
        plt.legend(fontsize=legend_font_size, labelspacing=0.1, ncol=2, columnspacing=0.1, bbox_to_anchor=(0, 1.02), loc='upper left', borderpad=0.1)
        # plt.legend(fontsize=legend_font_size,
        #            ncol=2,
        #            loc='upper left',
        #            bbox_to_anchor=(0, 1.2),
        #            columnspacing=0.3)

    plt.savefig(fig_save_path, bbox_inches='tight')
    plt.close()


# Average Bandwidth: Cohere + OPENAI
PLOT_AVERAGE_BANDWIDTH_COHERE = True
if PLOT_AVERAGE_BANDWIDTH_COHERE:
    # Data, set unused value to none
    fig_save_path = os.path.join(
        all_figure_dir, f'fig-15-{figure_name_prefix}-average-bandwidth-all.pdf')
    group_list = [
        'cohere-1m-concurrency-1', 'cohere-1m-concurrency-256',
        'cohere-10m-concurrency-1', 'cohere-10m-concurrency-256',
        'openai-500k-concurrency-1', 'openai-500k-concurrency-256',
        'openai-5m-concurrency-1', 'openai-5m-concurrency-256'
    ]
    y_values = {
        'cohere-1m-concurrency-1':
        results_average_bandwidth_avg['cohere-1m'][1],
        'cohere-1m-concurrency-256':
        results_average_bandwidth_avg['cohere-1m'][256],
        'cohere-10m-concurrency-1':
        results_average_bandwidth_avg['cohere-10m'][1],
        'cohere-10m-concurrency-256':
        results_average_bandwidth_avg['cohere-10m'][256],
        'openai-500k-concurrency-1':
        results_average_bandwidth_avg['openai-500k'][1],
        'openai-500k-concurrency-256':
        results_average_bandwidth_avg['openai-500k'][256],
        'openai-5m-concurrency-1':
        results_average_bandwidth_avg['openai-5m'][1],
        'openai-5m-concurrency-256':
        results_average_bandwidth_avg['openai-5m'][256],
    }
    std_dev = {
        'cohere-1m-concurrency-1':
        results_average_bandwidth_std['cohere-1m'][1],
        'cohere-1m-concurrency-256':
        results_average_bandwidth_std['cohere-1m'][256],
        'cohere-10m-concurrency-1':
        results_average_bandwidth_std['cohere-10m'][1],
        'cohere-10m-concurrency-256':
        results_average_bandwidth_std['cohere-10m'][256],
        'openai-500k-concurrency-1':
        results_average_bandwidth_std['openai-500k'][1],
        'openai-500k-concurrency-256':
        results_average_bandwidth_std['openai-500k'][256],
        'openai-5m-concurrency-1':
        results_average_bandwidth_std['openai-5m'][1],
        'openai-5m-concurrency-256':
        results_average_bandwidth_std['openai-5m'][256],
    }
    x_ticks = [str(x) for x in all_bwidth]

    title = None
    xlabel = 'Beam width'
    ylabel = 'Bandwidth (MiB/s)'

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
    ax.set_ylim(0, AVG_BANDWIDTH_MAX_Y)

    for (index, group_name) in zip(range(len(group_list)), group_list):
        # x, y, std_dev, data_label = data[group_name]
        x = range(1, len(y_values[group_name]) + 1)
        y = y_values[group_name]
        yerr = None
        if std_dev:
            yerr = std_dev[group_name]

        linestyle = 'solid'
        if group_name.endswith('256'):
            linestyle = 'dashed'

        if group_name == 'cohere-1m-concurrency-256':
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
    legend_group = [
        'Cohere 1M',
        'Cohere 10M',
        'OpenAI 500K',
        'OpenAI 5M',
    ]

    reset_color()
    for cur_group in legend_group:
        plt.plot([], [],
                 marker=dot_style[index % len(dot_style)],
                 linewidth=linewidth,
                 markersize=markersize,
                 color=get_next_color(),
                 label=cur_group)

    for cur_group in ['Concurrecy 1', 'Concurrency 256']:
        reset_color()
        plt.plot(
            [], [],
            linewidth=linewidth,
            linestyle='dashed' if cur_group == 'Concurrency 256' else 'solid',
            color=get_next_color(),
            label=cur_group)

    if True:
        plt.legend(fontsize=legend_font_size,
                   labelspacing=0.1,
                   ncol=2,
                   columnspacing=0.1,
                #    bbox_to_anchor=(0, 1.02),
                #    loc='upper left',
                #    borderpad=0.1
                   )
        # plt.legend(fontsize=legend_font_size,
        #            ncol=2,
        #            loc='upper left',
        #            bbox_to_anchor=(0, 1.2),
        #            columnspacing=0.3)


    plt.savefig(fig_save_path, bbox_inches='tight')
    plt.close()
