'''
The initial script for tracing the IO workload of Milvus with DiskANN
'''
import argparse
import os
import time
import signal
import subprocess
from datetime import datetime

import parse
import proc

'''
! run this command with sudo
sudo /home/zebin/anaconda3/envs/vectordb-bench-new/bin/python milvus-diskann-iotrace.py --case-type cohere-1m --concurrency 1 --run > milvus-diskann-iotrace-cohere-1m-con-1.log 2>&1
'''
'''
We have the following trace configurations:
- cohere 1m, concurrency = 1, 32, 256
- cohere 10m, concurrency = 1, 8, 256
- openai 500k, concurrency = 1, 16, 256
- openai 5m, concurrency = 1, 4, 256
'''

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

def create_non_exist_path(path_to_check):
    if not os.path.exists(path_to_check):
        os.makedirs(path_to_check, exist_ok=True)
        print(f"Created result directory: {path_to_check}")
    else:
        print(f"Result directory already exists: {path_to_check}")


vectordb_bench_path = os.getenv("VECTORDB_BENCH_BIN")
bpftrace_bin = os.getenv("BPFTRACE_BIN")

# Check if running as root at startup
bench_bin = f'{vectordb_bench_path} milvusdiskann --skip-drop-old --skip-load'
clean_page_cache_cmd = 'echo 1 | sudo tee /proc/sys/vm/drop_caches'
get_page_cache_cmd = 'grep -w nr_file_pages /proc/vmstat'
bio_trace_script = './bpf-scripts/bio-trace.bt'
result_path_root = 'results-io-trace'
figure_path_root = 'figures'

database_config = {
    'cohere-1m': {
        'uri': 'http://localhost:19539',
        'db-label': 'milvus_diskann_cohere_1m',
        'case-type': 'Performance768D1M',
        'searchlist': 10
    },
    'cohere-10m': {
        'uri': 'http://localhost:19540',
        'db-label': 'milvus_diskann_cohere_10m',
        'case-type': 'Performance768D10M',
        'searchlist': 10
    },
    'openai-500k': {
        'uri': 'http://localhost:19541',
        'db-label': 'milvus_diskann_openai_500k',
        'case-type': 'Performance1536D500K',
        'searchlist': 10
    },
    'openai-5m': {
        'uri': 'http://localhost:19542',
        'db-label': 'milvus_diskann_openai_5m',
        'case-type': 'Performance1536D5M',
        'searchlist': 10
    },
}

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
                    default=0,
                    help='Number of concurrent threads')
args = parser.parse_args()

RUN = args.run
if RUN:
    check_root()
assert args.case_type in [
    'cohere-1m', 'cohere-10m', 'openai-500k', 'openai-5m'
]
assert args.concurrency > 0
case_type = args.case_type
cur_config = database_config[args.case_type]
concurrency = args.concurrency

expr_results = os.path.join(result_path_root, f'milvus-diskann-{case_type}-con-{concurrency}')
expr_figures = os.path.join(figure_path_root, f'milvus-diskann-{case_type}-con-{concurrency}')
create_non_exist_path(expr_results)
create_non_exist_path(expr_figures)


# We only run the trace for one time, without repetitions
expr_str = f"{case_type}-con-{concurrency}-k-10-searchlist-{cur_config['searchlist']}"
vdb_bench_output_path = os.path.join(expr_results, f'vdb-bench-{expr_str}.log')
bio_trace_output_path = os.path.join(expr_results, f'bio-trace-{expr_str}.log')


# bio_trace_cmd = ['sudo', bpftrace_bin, bio_trace_script, ' > ', bio_trace_output_path]
bio_trace_cmd = ' '.join(
    ['sudo', bpftrace_bin, bio_trace_script, ' > ', bio_trace_output_path])
run_expr_cmd = f"{bench_bin} --case-type {cur_config['case-type']}  --db-label {cur_config['db-label']} --k 10 --num-concurrency {concurrency} --uri {cur_config['uri']} --password '' --search-list {cur_config['searchlist']}" + ' > ' + vdb_bench_output_path + ' 2>&1'

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

print('Execution finished, start parsing.')

## Parse the output
vectordb_perf = parse.parse_vectordb_bench_output(vdb_bench_output_path)
bitesize_file = bio_trace_output_path


## The I/O trace start before the actual experiments starts, so we compute the offset
trace_file = open(bitesize_file, 'r')
line = trace_file.readline().strip()
line = trace_file.readline().strip()  # skip the first line
trace_start = line.split()[-1]
vdb_start, vdb_end = parse.parse_experiment_times(vdb_bench_output_path)
fmt = "%H:%M:%S"
vdb_start_offset_in_trace = (
    datetime.strptime(vdb_start, fmt) -
    datetime.strptime(trace_start, fmt)).total_seconds()
vdb_end_offset_in_trace = (
    datetime.strptime(vdb_end, fmt) -
    datetime.strptime(trace_start, fmt)).total_seconds()

timestamps, op, bite_size, start_sector, num_sectors = parse.parse_bite_size(
    bitesize_file)  # timestamps is already in seconds
bite_size_by_op, bite_size_aggregated, io_size_by_sec = parse.process_bite_size(
    timestamps, op, bite_size)

print('vdb_start_offset_in_trace:', vdb_start_offset_in_trace)
print('vdb_end_offset_in_trace:', vdb_end_offset_in_trace)
print('read bandwidth:', io_size_by_sec['ALL_READS']['size'])
# print('Bite size aggregated:')
# for key in bite_size_aggregated:
#     print(key)
#     print('  ', bite_size_aggregated[key])
#     info_file.write(key + '\n')
#     info_file.write(str(bite_size_aggregated[key]) + '\n\n')

print(bite_size_aggregated.keys())

from plot import *

bitesize_save_prefix = f'bitesize-{expr_str}'
size_hist_op_prefix = f'bitesize-histogram-{expr_str}'

# Plot total bite size (Read)
if True:
    fig_save_path = f'total-read-{bitesize_save_prefix}.pdf'
    fig_save_path = os.path.join(expr_figures, fig_save_path)
    table_data = [['size (KB)', 'count']]
    call_count = bite_size_aggregated['ALL_READS']
    keys = sorted(call_count.keys())
    for cur_bitesize in keys:
        table_data.append([cur_bitesize, call_count[cur_bitesize]])

    fig, ax = plt.subplots()

    # Hide axes
    ax.axis("tight")
    ax.axis("off")

    # Create the table
    table = ax.table(cellText=table_data,
                     colLabels=None,
                     loc="right",
                     cellLoc='right')
    # Adjust the layout
    plt.subplots_adjust(left=0.2, top=0.8)

    # Show the table
    plt.savefig(fig_save_path, bbox_inches='tight')
    plt.close()

# Size histogram
size_hist_op = ['R', 'WS']
if True:
    for cur_op in size_hist_op:
        cur_bite_size_aggregated = bite_size_aggregated[cur_op]
        fig_save_path = f'{cur_op}-{bitesize_save_prefix}.pdf'
        fig_save_path = os.path.join(expr_figures, fig_save_path)

        # x
        x = sorted(cur_bite_size_aggregated.keys())
        # y
        y = [cur_bite_size_aggregated[cur_size] for cur_size in x]

        group_list = ['group1']
        x_values = {'group1': x}
        y_values = {'group1': y}
        std_dev = None
        x_ticks = None  # ['xtick_1', 'xtick_1']
        legend_label = {'group1': 'g1'}

        title = None
        xlabel = 'Bite size (KB)'
        ylabel = 'Count'

        # Parameters
        bar_width = 0.4

        datalabel_size = 26
        datalabel_va = 'bottom'
        axis_tick_font_size = 34
        axis_label_font_size = 44
        legend_font_size = 30

        # plot
        reset_color()
        fig, ax = plt.subplots(figsize=(12, 8))
        plt.grid(axis='y')  # x, y, both

        # set ticks
        if x_ticks:
            plt.xticks(list(np.arange(len(x_ticks))), x_ticks)

        if title:
            plt.title(title)

        if xlabel:
            plt.xlabel(xlabel, fontsize=axis_label_font_size)
        if ylabel:
            plt.ylabel(ylabel, fontsize=axis_label_font_size)

        ax.tick_params(axis='both',
                       which='major',
                       labelsize=axis_tick_font_size)

        # compute bar offset, with respect to center
        bar_offset = []
        mid_point = (len(group_list) * bar_width) / 2
        for i in range(len(group_list)):
            bar_offset.append(bar_width * i + 0.5 * bar_width - mid_point)

        # draw figure by column
        for (index, group_name) in zip(range(len(group_list)), group_list):
            x_axis = x_values['group1']
            y = y_values[group_name]
            yerr = None
            if std_dev:
                yerr = std_dev[group_name]
            bar_pos = x_axis  # + bar_offset[index]

            plt.bar(bar_pos,
                    y,
                    width=bar_width,
                    label=legend_label[group_name],
                    yerr=yerr,
                    color=get_next_color())

        # Legend: Change the ncol and loc to fine-tune the location of legend
        if legend_label != None:
            plt.legend(fontsize=legend_font_size)

        plt.savefig(fig_save_path, bbox_inches='tight')
        plt.close()

# Plot bite size: bandwidth (read)
if True:
    fig_save_path = f'per-sec-readwrite-{bitesize_save_prefix}.pdf'
    fig_save_path = os.path.join(expr_figures, fig_save_path)
    group_list = ['read', 'write']
    y_values = {
        'read': io_size_by_sec['ALL_READS']['size'],
        'write': io_size_by_sec['ALL_WRITES']['size'],
    }
    y_values['read'] = [x / 1024 for x in y_values['read']]
    y_values['write'] = [x / 1024 for x in y_values['write']]
    std_dev = None
    # x_ticks = ['xtick_1', 'xtick_1']
    legend_label = {'read': 'read', 'write': 'write'}

    title = None
    xlabel = 'Time (S)'
    ylabel = 'Bandwidth (GiB/s)'

    # Parameters
    linewidth = 4
    markersize = 15

    datalabel_size = 26
    datalabel_va = 'bottom'
    axis_tick_font_size = 45
    axis_label_font_size = 55
    legend_font_size = 45

    reset_color()
    fig, ax = plt.subplots(figsize=(12, 8))

    plt.xlabel(xlabel, fontsize=axis_label_font_size)
    plt.ylabel(ylabel, fontsize=axis_label_font_size)
    plt.grid(True)

    ax.tick_params(axis='both', which='major', labelsize=axis_tick_font_size)

    # Ensuring timing is equal for both lines
    time_min = min(io_size_by_sec['ALL_READS']['ts'][0],
                   io_size_by_sec['ALL_WRITES']['ts'][0])
    time_max = max(io_size_by_sec['ALL_READS']['ts'][-1],
                   io_size_by_sec['ALL_WRITES']['ts'][-1])
    ran = time_max - time_min + 1

    print(io_size_by_sec['ALL_READS'])

    if (io_size_by_sec['ALL_READS']['ts'][0] > time_min):
        y_values['read'] = [0] * (io_size_by_sec['ALL_READS']['ts'][0] -
                                  time_min) + y_values['read']
    if (io_size_by_sec['ALL_READS']['ts'][-1] < time_max):
        y_values['read'] = y_values['read'] + [0] * (
            time_max - io_size_by_sec['ALL_READS']['ts'][-1])

    if (io_size_by_sec['ALL_WRITES']['ts'][0] > time_min):
        y_values['write'] = [0] * (io_size_by_sec['ALL_WRITES']['ts'][0] -
                                   time_min) + y_values['write']
    if (io_size_by_sec['ALL_WRITES']['ts'][-1] < time_max):
        y_values['write'] = y_values['write'] + [0] * (
            time_max - io_size_by_sec['ALL_WRITES']['ts'][-1])

    print(y_values['read'])
    for (index, group_name) in zip(range(len(group_list)), group_list):
        # x, y, std_dev, data_label = data[group_name]
        x = range(1, len(y_values[group_name]) + 1)
        y = y_values[group_name]

        yerr = None
        if std_dev:
            yerr = std_dev[group_name]

        # TODO: Add this to the github plot repo
        if legend_label == None:
            cur_legend_label = 'placeholder'
        else:
            cur_legend_label = legend_label[group_name]

        plt.errorbar(
            x,
            [float(yy) / 1024 for yy in y],
            yerr=yerr,
            label=cur_legend_label,
            # marker=dot_style[index % len(dot_style)],
            linewidth=linewidth,
            markersize=markersize,
            color=get_next_color(),
        )
        # ax.set_xlim(0, 60)
        # ax.set_xticks(range(0, 300, 60))
        # ax.set_xticklabels([str(size // 60) for size in range(0, 300, 60)])
        # ax.set_ylim(0, 10)
        # ax.set_yticks(range(1, 11))

    # Add vertical lines at experiment start and end
    plt.axvline(x=vdb_start_offset_in_trace, color='blue', linestyle='--', linewidth=3) #, label='Query Start')
    plt.axvline(x=vdb_end_offset_in_trace, color='black', linestyle='--', linewidth=3) #, label='Query End')

    if legend_label != None:
        plt.legend(loc='lower right', prop={'size': 54}, labelspacing=0.1)

    plt.savefig(fig_save_path, bbox_inches='tight')
    plt.close()
