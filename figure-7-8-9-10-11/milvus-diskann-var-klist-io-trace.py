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

import parse
import proc

def create_non_exist_path(path_to_check):
    if not os.path.exists(path_to_check):
        os.makedirs(path_to_check, exist_ok=True)
        print(f"Created result directory: {path_to_check}")
    else:
        print(f"Result directory already exists: {path_to_check}")


'''
Example command:
sudo /home/zebin/anaconda3/envs/vectordb-bench-new/bin/python milvus-diskann-var-klist-io-trace.py --case-type cohere-1m --concurrency 1 --run > milvus-var-klist-diskann-cohere-1m-concurrency-1-io-trace.log 2>&1 &
nohup python3 milvus-diskann-var-klist-io-trace.py --case-type cohere-10m --concurrency 1 --run > milvus-var-klist-diskann-cohere-10m-concurrency-1-io-trace.log 2>&1 &
nohup python3 milvus-diskann-var-klist-io-trace.py --case-type openai-500k --concurrency 1 --run > milvus-var-klist-diskann-openai-500k-concurrency-1-io-trace.log 2>&1 &
nohup python3 milvus-diskann-var-klist-io-trace.py --case-type openai-5m --concurrency 1 --run > milvus-var-klist-diskann-openai-5m-concurrency-1-io-trace.log 2>&1 & 
'''

vectordb_bench_path = os.getenv("VECTORDB_BENCH_BIN")
bpftrace_bin = os.getenv("BPFTRACE_BIN")

DEBUG = False
DEBUG_MIN = False
bench_bin = f'{vectordb_bench_path} milvusdiskann --skip-drop-old --skip-load --skip-search-serial'
clean_page_cache_cmd = 'echo 1 | sudo tee /proc/sys/vm/drop_caches'
get_page_cache_cmd = 'grep -w nr_file_pages /proc/vmstat'
bpftrace_bin = f'{bpftrace_bin}'
bio_trace_script = './bpf-scripts/bio-trace.bt'
result_path_root = 'results-io-trace-var-arg'
figure_path_root = 'figures'


def check_root():
    """Check if the script is running with root privileges"""
    if os.geteuid() != 0:
        print("Error: This script must be run as root (sudo).")
        print(
            "Please run with: sudo /home/zebin/anaconda3/envs/vectordb-bench-new/bin/python diskann-trace.py"
        )
        exit(1)
    else:
        print("Running with root privileges.")

# variables
all_searchlist = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
rep = 5

# Debug set
if DEBUG:
    if DEBUG_MIN:
        all_searchlist = [10]
        rep = 1
    else:
        all_searchlist = [10, 20]
        rep = 2

# We only allow these pre-defined arguments
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

# Parser input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--run',
                    action='store_true',
                    default=False,
                    help='Execute the commands')
parser.add_argument('--case-type',
                    type=str,
                    default='',
                    help='Case type for the test')
parser.add_argument('--concurrency',
                    type=int,
                    default=1,
                    help='Number of concurrency for the test')
args = parser.parse_args()

RUN = args.run
assert args.case_type in [
    'cohere-1m', 'cohere-10m', 'openai-500k', 'openai-5m'
]
case_type = args.case_type
cur_config = database_config[args.case_type]
concurrency = args.concurrency


# Check if the result path root exists, create it if it doesn't
expr_results_common = os.path.join(result_path_root, f'milvus-diskann-{case_type}-var-klist-concurrency-{concurrency}-io-trace')
expr_results_path = os.path.join(
    expr_results_common,
    f'milvus-diskann-{case_type}-var-klist-concurrency-{concurrency}-output')
bio_trace_path = os.path.join(
    expr_results_common,
    f'milvus-diskann-{case_type}-var-klist-concurrency-{concurrency}-io-trace'
)


create_non_exist_path(expr_results_path)
create_non_exist_path(bio_trace_path)

for cur_searchlist in all_searchlist:
    for cur_rep in range(rep):
        # Trace CPU, memory and page cache
        expr_str = f"{case_type}-concurrency-{concurrency}-k-10-searchlist-{cur_searchlist}-rep-{cur_rep}.log"
        bio_trace_output_path = os.path.join(bio_trace_path, 'io-trace-' + expr_str)
        bio_trace_cmd = ' '.join([
            'sudo', bpftrace_bin, bio_trace_script, ' > ',
            bio_trace_output_path
        ])

        # experiment command
        vdb_output_file = f'vdb-bench-{expr_str}'
        vdb_bench_output_path = os.path.join(expr_results_path, vdb_output_file)

        # Note that we do not allow index build in this scrip to prevent dropping the database by mistake
        run_expr_cmd = f"{bench_bin} --case-type {cur_config['case-type']}  --db-label {cur_config['db-label']} --k 10 --num-concurrency {concurrency} --uri {cur_config['uri']} --password '' --search-list {cur_searchlist}"
        run_expr_cmd = run_expr_cmd + ' > ' + vdb_bench_output_path + ' 2>&1'

        print('Clean page cache, number of pages before clean:', end = '')
        proc.exec_cmd(get_page_cache_cmd, RUN)  # get page cache before clean
        proc.exec_cmd('sync', RUN)
        proc.exec_cmd(clean_page_cache_cmd, RUN) # clean page cache before each run
        print('Clean page cache, number of pages after clean:', end = '')
        proc.exec_cmd(get_page_cache_cmd, RUN)
        p_bio_trace = proc.exec_cmd_background(bio_trace_cmd, RUN)
        proc.exec_cmd(run_expr_cmd, RUN)

        if RUN:
            # subprocess.run(["sudo", "kill", "-TERM", f"-{p_bio_trace.pid}"],
            #    check=True)
            os.killpg(os.getpgid(p_bio_trace.pid), signal.SIGKILL)
            time.sleep(5)


## Parse
# Performance results
results_qps = {}
results_latency = {}
results_latency_con_avg = {}
results_latency_con_p99 = {}
results_recall = {}
results_qps_average = []
results_latency_average = []
results_latency_con_avg_average = []
results_latency_con_p99_average = []
results_recall_average = []
results_qps_std = []
results_latency_std = []
results_latency_con_avg_std = []
results_latency_con_p99_std = []
results_recall_std = []


# Trace results
results_start_end_offset = {}
results_bandwidth_trace = {}
results_total_bandwidth_avg = []
results_average_bandwidth_avg = []
results_total_bandwidth_std = []
results_average_bandwidth_std = []


for cur_searchlist in all_searchlist:
    results_qps[cur_searchlist] = []
    results_latency[cur_searchlist] = []
    results_recall[cur_searchlist] = []
    results_latency_con_avg[cur_searchlist] = []
    results_latency_con_p99[cur_searchlist] = []

    for cur_rep in range(rep):
        cur_rep_total_bandwidth = []
        cur_rep_average_bandwidth = []
        # Trace CPU, memory and page cache
        expr_str = f"{case_type}-concurrency-{concurrency}-k-10-searchlist-{cur_searchlist}-rep-{cur_rep}.log"
        vdb_output_file = f'vdb-bench-{expr_str}'
        vdb_bench_output_path = os.path.join(expr_results_path, vdb_output_file)
        bio_trace_output_path = os.path.join(bio_trace_path, 'io-trace-' + expr_str)


        # print for debug
        print(f"Parsing: searchlist {cur_searchlist}, rep {cur_rep}")

        # Parse the performance
        vectordb_perf = parse.parse_vectordb_bench_output(vdb_bench_output_path)

        results_qps[cur_searchlist].append(vectordb_perf['qps'])
        results_latency[cur_searchlist].append(vectordb_perf['latency'])
        results_latency_con_avg[cur_searchlist].append(vectordb_perf[
            'con_latency_avg'])
        results_latency_con_p99[cur_searchlist].append(vectordb_perf['con_latency_p99'])
        results_recall[cur_searchlist].append(vectordb_perf['recall'])

        # Parse the trace
        vdb_start, vdb_end = parse.parse_experiment_times(vdb_bench_output_path)
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
            bio_trace_output_path)  # timestamps is already in seconds
        bite_size_by_op, bite_size_aggregated, io_size_by_sec = parse.process_bite_size(
            timestamps, op, bite_size)
        cur_read_bandwidth_trace = [x/1024 for x in io_size_by_sec['ALL_READS']['size']]
        if rep == 0:
            # for the traces, we only add rep = 0
            results_start_end_offset[cur_searchlist] = (vdb_start_offset_in_trace, vdb_end_offset_in_trace)
            results_bandwidth_trace[cur_searchlist] = cur_read_bandwidth_trace

        # print(f"vdb_start: {vdb_start}, vdb_end: {vdb_end}, trace_start: {trace_start}")
        # print(f"vdb_start_offset_in_trace: {vdb_start_offset_in_trace}, vdb_end_offset_in_trace: {vdb_end_offset_in_trace}")
        # print(f"cur_read_bandwidth_trace length: {len(cur_read_bandwidth_trace)}")
        # print(f"cur_bandwidth trace: {cur_read_bandwidth_trace}")
        # get the average and per-qps bandwidth
        clipped_bandwidth = cur_read_bandwidth_trace[int(vdb_start_offset_in_trace):int(vdb_end_offset_in_trace)+1]
        cur_total_bandwidth = sum(clipped_bandwidth) / len(clipped_bandwidth)
        cur_average_bandwidth = cur_total_bandwidth / vectordb_perf['qps']
        cur_rep_total_bandwidth.append(cur_total_bandwidth)
        cur_rep_average_bandwidth.append(cur_average_bandwidth)


    # performance
    results_qps_average.append(
        sum(results_qps[cur_searchlist]) / len(results_qps[cur_searchlist]))
    results_latency_average.append(
        sum(results_latency[cur_searchlist]) / len(results_latency[cur_searchlist]))
    results_latency_con_avg_average.append(
        sum(results_latency_con_avg[cur_searchlist]) /
        len(results_latency_con_avg[cur_searchlist]))
    results_latency_con_p99_average.append(
        sum(results_latency_con_p99[cur_searchlist]) /
        len(results_latency_con_p99[cur_searchlist]))
    results_recall_average.append(
        sum(results_recall[cur_searchlist]) / len(results_recall[cur_searchlist]))
    results_qps_std.append(float(np.std(results_qps[cur_searchlist])))
    results_latency_std.append(float(np.std(results_latency[cur_searchlist])))
    results_latency_con_avg_std.append(
        float(np.std(results_latency_con_avg[cur_searchlist])))
    results_latency_con_p99_std.append(
        float(np.std(results_latency_con_p99[cur_searchlist])))
    results_recall_std.append(float(np.std(results_recall[cur_searchlist])))

    # bandwidth
    results_total_bandwidth_avg.append(sum(cur_rep_total_bandwidth) /
                                       len(cur_rep_total_bandwidth))
    results_average_bandwidth_avg.append(
        sum(cur_rep_average_bandwidth) / len(cur_rep_average_bandwidth))
    results_total_bandwidth_std.append(float(np.std(cur_rep_total_bandwidth)))
    results_average_bandwidth_std.append(
        float(np.std(cur_rep_average_bandwidth)))

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

# Print with argument values for clarity
print("\nDetailed Results:")
for i, val in enumerate(all_searchlist):
    print(f"searchlist={val}:")
    print(f"  QPS: {results_qps_average[i]:.2f} ± {results_qps_std[i]:.2f}")
    print(f"  Latency: {results_latency_average[i]:.2f} ± {results_latency_std[i]:.2f}")
    print(f"  Recall: {results_recall_average[i]:.2f} ± {results_recall_std[i]:.2f}")
    print(f"  Total Bandwidth: {results_total_bandwidth_avg[i]:.2f} ± {results_total_bandwidth_std[i]:.2f} MiB/s")
    print(f"  Average Bandwidth: {results_average_bandwidth_avg[i]:.2f} ± {results_average_bandwidth_std[i]:.2f} MiB/s")

# Plot
from plot import *

PLOT_QPS = True
PLOT_LATENCY = True
PLOT_RECALL = True

figure_name_prefix = f'milvus-diskann-var-klist-{case_type}-concurrency-{concurrency}'
all_figure_dir = os.path.join(figure_path_root, f'milvus-diskann-var-klist-{case_type}-concurrency-{concurrency}')
create_non_exist_path(all_figure_dir)

if PLOT_QPS:
    # Data, set unused value to none
    fig_save_path = os.path.join(all_figure_dir, f'{figure_name_prefix}-qps.pdf')
    group_list = ['default']
    y_values = {'default': results_qps_average}
    std_dev = {'default': results_qps_std}
    x_ticks = [str(x) for x in all_searchlist]
    legend_label = {'default': 'default'}

    title = None
    xlabel = 'Search list size'
    ylabel = 'QPS'

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
    # ax.set_ylim()

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

if PLOT_LATENCY:
    # Data, set unused value to none
    fig_save_path = os.path.join(all_figure_dir, f'{figure_name_prefix}-latency.pdf')
    group_list = ['default']
    y_values = {'default': results_latency_average}
    std_dev = {'default': results_latency_std}
    x_ticks = [str(x) for x in all_searchlist]
    legend_label = {'default': 'default'}

    title = None
    xlabel = 'klist'
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
    # ax.set_xlim()
    # ax.set_ylim()

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

PLOT_TOTAL_BANDWIDTH = True
if PLOT_TOTAL_BANDWIDTH:
    # Data, set unused value to none
    fig_save_path = os.path.join(all_figure_dir,
                                 f'{figure_name_prefix}-total-bandwidth.pdf')
    group_list = ['default']
    y_values = {'default': results_total_bandwidth_avg}
    std_dev = {'default': results_total_bandwidth_std}
    x_ticks = [str(x) for x in all_searchlist]
    legend_label = {'default': 'default'}

    title = None
    xlabel = 'klist'
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
    # ax.set_xlim()
    # ax.set_ylim()

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

PLOT_AVERAGE_BANDWIDTH = True
if PLOT_AVERAGE_BANDWIDTH:
    # Data, set unused value to none
    fig_save_path = os.path.join(all_figure_dir,
                                 f'{figure_name_prefix}-average-bandwidth.pdf')
    group_list = ['default']
    y_values = {'default': results_average_bandwidth_avg}
    std_dev = {'default': results_average_bandwidth_std}
    x_ticks = [str(x) for x in all_searchlist]
    legend_label = {'default': 'default'}

    title = None
    xlabel = 'klist'
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
    # ax.set_xlim()
    # ax.set_ylim()

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

PLOT_LATENCY_AVERAGE = False
if PLOT_LATENCY_AVERAGE:
    # Data, set unused value to none
    fig_save_path = os.path.join(all_figure_dir, f'{figure_name_prefix}-latency-avg.pdf')
    group_list = ['default']
    y_values = {'default': results_latency_con_avg_average}
    std_dev = {'default': results_latency_con_avg_std}
    x_ticks = [str(x) for x in all_searchlist]
    legend_label = {'default': 'default'}

    title = None
    xlabel = 'klist'
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
    # ax.set_xlim()
    # ax.set_ylim()

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

PLOT_LATENCY_P99 = False
if PLOT_LATENCY_P99:
    # Data, set unused value to none
    fig_save_path = os.path.join(all_figure_dir,
                                 f'{figure_name_prefix}-latency-p99.pdf')
    group_list = ['default']
    y_values = {'default': results_latency_con_p99_average}
    std_dev = {'default': results_latency_con_p99_std}
    x_ticks = [str(x) for x in all_searchlist]
    legend_label = {'default': 'default'}

    title = None
    xlabel = 'klist'
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
    # ax.set_xlim()
    # ax.set_ylim()

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

PLOT_ACC = False
if PLOT_ACC:
    # Data, set unused value to none
    fig_save_path = os.path.join(all_figure_dir,
                                 f'{figure_name_prefix}-recall.pdf')
    group_list = ['default']
    y_values = {'default': results_recall_average}
    std_dev = {'default': results_recall_std}
    x_ticks = [str(x) for x in all_searchlist]
    legend_label = {'default': 'default'}

    title = None
    xlabel = 'Search list size'
    ylabel = 'QPS'

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
    # ax.set_ylim()

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

PLOT_TRACE = False
if PLOT_TRACE:
    for cur_searchlist in all_searchlist:
        # Plot CPU trace
        global_cpu_idle = results_cpu_trace[cur_searchlist]['idle']
        global_cpu_usage = [100 - i for i in global_cpu_idle]

        fig_save_path = os.path.join(
            all_figure_dir,
            f'{figure_name_prefix}-cpu-concurrency-{cur_searchlist}.pdf'
        )
        group_list = ['default']
        y_values = {'default': global_cpu_usage}
        std_dev = None
        x_ticks = [str(i) for i in range(len(y_values['default']))]
        legend_label = {'default': 'default'}

        title = None
        xlabel = 'Time (s)'
        ylabel = 'CPU usage'

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
        # Only plot grid if we have visible ticks
        if len(ax.xaxis.get_ticklabels()) > 0 and len(
                ax.yaxis.get_ticklabels()) > 0:
            plt.grid(True)

        ax.tick_params(axis='both',
                       which='major',
                       labelsize=axis_tick_font_size)
        # ax.xaxis.set_ticks(range(len(x_ticks)))
        # # ax.set_xticklabels(x_ticks)
        # # Only show every 5th tick on x-axis for better readability
        # every_nth = 5
        # for n, label in enumerate(ax.xaxis.get_ticklabels()):
        #     if n % every_nth != 0 and n != 0:
        #         label.set_visible(False)
        ax.set_xlim(0, 60)
        # ax.set_ylim()

        for (index, group_name) in zip(range(len(group_list)), group_list):
            # x, y, std_dev, data_label = data[group_name]
            x = range(len(y_values[group_name]))
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

        # Plot memory trace
        global_mem_usage = results_mem_trace[cur_searchlist]['used']

        fig_save_path = os.path.join(
            all_figure_dir,
            f'{figure_name_prefix}-mem-concurrency-{cur_searchlist}.pdf')
        group_list = ['default']
        y_values = {'default': global_mem_usage}
        std_dev = None
        x_ticks = [str(i) for i in range(len(y_values['default']))]
        legend_label = {'default': 'default'}

        title = None
        xlabel = 'Time (s)'
        ylabel = 'Mem usage (MiB)'

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
        # Only plot grid if we have visible ticks
        if len(ax.xaxis.get_ticklabels()) > 0 and len(
                ax.yaxis.get_ticklabels()) > 0:
            plt.grid(True)

        ax.tick_params(axis='both',
                       which='major',
                       labelsize=axis_tick_font_size)
        # ax.xaxis.set_ticks(range(len(x_ticks)))
        # # ax.set_xticklabels(x_ticks)
        # # Only show every 5th tick on x-axis for better readability
        # every_nth = 5
        # for n, label in enumerate(ax.xaxis.get_ticklabels()):
        #     if n % every_nth != 0 and n != 0:
        #         label.set_visible(False)
        ax.set_xlim(0, 60)
        # ax.set_ylim()

        for (index, group_name) in zip(range(len(group_list)), group_list):
            # x, y, std_dev, data_label = data[group_name]
            x = range(len(y_values[group_name]))
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

        # Plot page cache trace
        fig_save_path = os.path.join(all_figure_dir, f'{figure_name_prefix}-page-cache-concurrency-{cur_searchlist}.pdf')
        group_list = ['default']
        y_values = {'default': results_page_cache_trace[cur_searchlist]['nr_file_pages']}
        std_dev = None
        x_ticks = [str(i) for i in range(len(y_values['default']))]
        legend_label = {'default': 'default'}

        title = None
        xlabel = 'Time (s)'
        ylabel = 'Cached pages'

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
        # Only plot grid if we have visible ticks
        if len(ax.xaxis.get_ticklabels()) > 0 and len(ax.yaxis.get_ticklabels()) > 0:
            plt.grid(True)

        ax.tick_params(axis='both', which='major', labelsize=axis_tick_font_size)
        # ax.xaxis.set_ticks(range(len(x_ticks)))
        # # ax.set_xticklabels(x_ticks)
        # # Only show every 5th tick on x-axis for better readability
        # every_nth = 5
        # for n, label in enumerate(ax.xaxis.get_ticklabels()):
        #     if n % every_nth != 0 and n != 0:
        #         label.set_visible(False)
        ax.set_xlim(0, 60)
        # ax.set_ylim()

        for (index, group_name) in zip(range(len(group_list)), group_list):
            # x, y, std_dev, data_label = data[group_name]
            x = range(len(y_values[group_name]))
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
