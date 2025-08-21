'''
The initial script for tracing the IO workload of Milvus with DiskANN
'''
import os
import numpy as np
from datetime import datetime
import pickle

import parse

from utils import create_non_exist_path

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

results_save_path = 'all-io-bandwidth-parsed.bin'

vector_dbs = ['milvus']
vdb_configs = {
    'milvus': ['diskann'],
}
datasets = ['cohere-1m', 'cohere-10m', 'openai-500k', 'openai-5m']
expr_config_fname = {
    'milvus': {
        'diskann-cohere-1m': 'k-10-searchlist-10',
        'diskann-cohere-10m': 'k-10-searchlist-10',
        'diskann-openai-500k': 'k-10-searchlist-10',
        'diskann-openai-5m': 'k-10-searchlist-10',
    },
}

all_concurrency = {
    'milvus': {
        'diskann-cohere-1m': [1, 32, 256],
        'diskann-cohere-10m': [1, 8, 256],
        'diskann-openai-500k': [1, 16, 256],
        'diskann-openai-5m': [1, 4, 256],
    },
}

result_path_root = 'io-trace'
figure_path_root = 'figures'

if os.path.exists(results_save_path):
    with open(results_save_path, 'rb') as f:
        bandwidth_trace = pickle.load(f)
        start_and_end_trace = pickle.load(f)
        qps = pickle.load(f)

    print('Loaded existing results from:', results_save_path)
    print('Skip parsing, delete thi cache file to re-parse the results')
else:
    bandwidth_trace = {}
    start_and_end_trace = {}
    qps = {}
    for cur_dataset in datasets:
        bandwidth_trace[cur_dataset] = {}
        start_and_end_trace[cur_dataset] = {}
        qps[cur_dataset] = {}
        for cur_vdb in vector_dbs:
            for cur_config in vdb_configs[cur_vdb]:
                for cur_concurrency in all_concurrency[cur_vdb][f'{cur_config}-{cur_dataset}']:
                    cur_line_name = f'{cur_vdb}-{cur_config}-con-{cur_concurrency}'

                    # output name
                    expr_results = os.path.join(result_path_root, f'milvus-diskann-{cur_dataset}-con-{cur_concurrency}')
                    expr_str = f"{cur_dataset}-con-{cur_concurrency}-{expr_config_fname[cur_vdb][cur_config +'-'+cur_dataset]}"
                    vdb_bench_output_path = os.path.join(expr_results, f'vdb-bench-{expr_str}.log')
                    bio_trace_output_path = os.path.join(expr_results, f'bio-trace-{expr_str}.log')

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

                    bandwidth_trace[cur_dataset][cur_line_name] = io_size_by_sec
                    start_and_end_trace[cur_dataset][cur_line_name] = (vdb_start_offset_in_trace,
                                                                    vdb_end_offset_in_trace)
                    qps[cur_dataset][cur_line_name] = vectordb_perf['qps']/1000

    with open(results_save_path, 'wb') as f:
        pickle.dump(bandwidth_trace, f)
        pickle.dump(start_and_end_trace, f)
        pickle.dump(qps, f)

# print('Bite size aggregated:')
# for key in bite_size_aggregated:
#     print(key)
#     print('  ', bite_size_aggregated[key])
#     info_file.write(key + '\n')
#     info_file.write(str(bite_size_aggregated[key]) + '\n\n')

# print(bite_size_aggregated.keys())

# Global x/y limits
bandwidth_y_max = 1000

from plot import *

# bitesize_save_prefix = f'bitesize-{expr_str}'
# size_hist_op_prefix = f'bitesize-histogram-{expr_str}'

fig_save_path_all = f'{figure_path_root}'
create_non_exist_path(fig_save_path_all)

average_bandwidth = {}
average_bandwidth_per_req = {}
end_extension = 5


PLOT_COHERE_1M = True
if PLOT_COHERE_1M:
    average_bandwidth['cohere-1m'] = {}
    average_bandwidth_per_req['cohere-1m'] = {}
    fig_save_path = os.path.join(fig_save_path_all, 'fig-5-a-rw-bandwidth-cohere-1m.pdf')
    # group_list = ['read', 'write']
    # y_values = {
    #     'read': io_size_by_sec['ALL_READS']['size'],
    #     'write': io_size_by_sec['ALL_WRITES']['size'],
    # }

    # currently we only plot read bandwidth
    group_list = bandwidth_trace['cohere-1m'].keys()
    y_values = {}
    for cur_group in group_list:
        cur_start, cur_end = start_and_end_trace['cohere-1m'][cur_group]
        print(f"Processing group: {cur_group}, start: {cur_start}, end: {cur_end}")
        print(
            f"y_values[{cur_group}], {len(bandwidth_trace['cohere-1m'][cur_group]['ALL_READS']['size'])}: {bandwidth_trace['cohere-1m'][cur_group]['ALL_READS']['size']}"
        )
        cur_start = int(cur_start)
        cur_end = int(cur_end) + end_extension
        y_values[cur_group] = bandwidth_trace['cohere-1m'][cur_group]['ALL_READS']['size'][cur_start: cur_end+1] # in MiB
        # print(f"y_values[{cur_group}], {len(y_values[cur_group])}: {y_values[cur_group]}")
        cur_average_bandwidth = sum(y_values[cur_group]) / len(y_values[cur_group])
        average_bandwidth['cohere-1m'][cur_group] = cur_average_bandwidth
        average_bandwidth_per_req['cohere-1m'][cur_group] = cur_average_bandwidth / (
                qps['cohere-1m'][cur_group] * 1000)
    std_dev = None
    # x_ticks = ['xtick_1', 'xtick_1']
    legend_label = {
        'milvus-diskann-con-1': '1 thread',
        'milvus-diskann-con-32': '32 threads',
        'milvus-diskann-con-256': '256 threads',
    }

    title = None
    xlabel = 'Time (S)'
    ylabel = 'Bandwidth (MiB/s)'

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

    for (index, group_name) in zip(range(len(group_list)), group_list):
        # Ensuring timing is equal for both lines
        io_size_by_sec = bandwidth_trace['cohere-1m'][group_name]
        time_min = min(io_size_by_sec['ALL_READS']['ts'][0],
                       io_size_by_sec['ALL_WRITES']['ts'][0])
        time_max = max(io_size_by_sec['ALL_READS']['ts'][-1],
                       io_size_by_sec['ALL_WRITES']['ts'][-1])
        ran = time_max - time_min + 1

        # print(io_size_by_sec['ALL_READS'])
        # print(y_values[group_name])

        #### Currently, we only print read, so no need to show the write part
        # if (io_size_by_sec['ALL_READS']['ts'][0] > time_min):
        #     y_values['read'] = [0] * (io_size_by_sec['ALL_READS']['ts'][0] -
        #                             time_min) + y_values['read']
        # if (io_size_by_sec['ALL_READS']['ts'][-1] < time_max):
        #     y_values['read'] = y_values['read'] + [0] * (
        #         time_max - io_size_by_sec['ALL_READS']['ts'][-1])

        # if (io_size_by_sec['ALL_WRITES']['ts'][0] > time_min):
        #     y_values['write'] = [0] * (io_size_by_sec['ALL_WRITES']['ts'][0] -
        #                             time_min) + y_values['write']
        # if (io_size_by_sec['ALL_WRITES']['ts'][-1] < time_max):
        #     y_values['write'] = y_values['write'] + [0] * (
        #         time_max - io_size_by_sec['ALL_WRITES']['ts'][-1])

        # x, y, std_dev, data_label = data[group_name]
        x = range(0, len(y_values[group_name]))
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
    ax.set_xlim(0, 30+end_extension)
    # ax.set_xticks(range(0, 300, 60))
    # ax.set_xticklabels([str(size // 60) for size in range(0, 300, 60)])
    ax.set_ylim(0, bandwidth_y_max)
    # Set y-ticks at 0, 250, 500, 750, and 1000
    ax.set_yticks([0, 250, 500, 750, 1000])
    ax.set_yticklabels(['0', '250', '500', '750', '1,000'])

    # # Add vertical lines at experiment start and end
    # plt.axvline(x=vdb_start_offset_in_trace,
    #             color='blue',
    #             linestyle='--',
    #             linewidth=3)  #, label='Query Start')
    # plt.axvline(x=vdb_end_offset_in_trace,
    #             color='black',
    #             linestyle='--',
    #             linewidth=3)  #, label='Query End')

    # Add vertical line at x=20
    plt.axvline(x=30, color='blue', linestyle='--', linewidth=3)
    if legend_label != None:
        # plt.legend(fontsize=legend_font_size, labelspacing=0.1)
        plt.legend(
            loc='upper left',
            fontsize=legend_font_size,
            labelspacing=0.1,
        )

    plt.savefig(fig_save_path, bbox_inches='tight')
    plt.close()


PLOT_COHERE_10M = True
if PLOT_COHERE_10M:
    average_bandwidth['cohere-10m'] = {}
    average_bandwidth_per_req['cohere-10m'] = {}
    fig_save_path = os.path.join(fig_save_path_all,
                                 'fig-5-a-rw-bandwidth-cohere-10m.pdf')
    # group_list = ['read', 'write']
    # y_values = {
    #     'read': io_size_by_sec['ALL_READS']['size'],
    #     'write': io_size_by_sec['ALL_WRITES']['size'],
    # }

    # currently we only plot read bandwidth
    group_list = bandwidth_trace['cohere-10m'].keys()
    y_values = {}

    for cur_group in group_list:
        cur_start, cur_end = start_and_end_trace['cohere-10m'][cur_group]
        cur_start = int(cur_start)
        cur_end = int(cur_end) + end_extension
        y_values[cur_group] = bandwidth_trace['cohere-10m'][cur_group]['ALL_READS']['size'][cur_start: cur_end+1]
        cur_average_bandwidth = sum(y_values[cur_group]) / len(
            y_values[cur_group])
        average_bandwidth['cohere-10m'][cur_group] = cur_average_bandwidth
        average_bandwidth_per_req['cohere-10m'][
            cur_group] = cur_average_bandwidth / (qps['cohere-10m'][cur_group] *
                                                  1000)
    std_dev = None
    # x_ticks = ['xtick_1', 'xtick_1']
    legend_label = {
        'milvus-diskann-con-1': '1 thread',
        'milvus-diskann-con-8': '8 threads',
        'milvus-diskann-con-256': '256 threads',
    }

    title = None
    xlabel = 'Time (S)'
    ylabel = 'Bandwidth (MiB/s)'

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

    for (index, group_name) in zip(range(len(group_list)), group_list):
        # Ensuring timing is equal for both lines
        io_size_by_sec = bandwidth_trace['cohere-10m'][group_name]
        time_min = min(io_size_by_sec['ALL_READS']['ts'][0],
                    io_size_by_sec['ALL_WRITES']['ts'][0])
        time_max = max(io_size_by_sec['ALL_READS']['ts'][-1],
                    io_size_by_sec['ALL_WRITES']['ts'][-1])
        ran = time_max - time_min + 1

        # print(io_size_by_sec['ALL_READS'])
        # print(y_values[group_name])

        #### Currently, we only print read, so no need to show the write part
        # if (io_size_by_sec['ALL_READS']['ts'][0] > time_min):
        #     y_values['read'] = [0] * (io_size_by_sec['ALL_READS']['ts'][0] -
        #                             time_min) + y_values['read']
        # if (io_size_by_sec['ALL_READS']['ts'][-1] < time_max):
        #     y_values['read'] = y_values['read'] + [0] * (
        #         time_max - io_size_by_sec['ALL_READS']['ts'][-1])

        # if (io_size_by_sec['ALL_WRITES']['ts'][0] > time_min):
        #     y_values['write'] = [0] * (io_size_by_sec['ALL_WRITES']['ts'][0] -
        #                             time_min) + y_values['write']
        # if (io_size_by_sec['ALL_WRITES']['ts'][-1] < time_max):
        #     y_values['write'] = y_values['write'] + [0] * (
        #         time_max - io_size_by_sec['ALL_WRITES']['ts'][-1])

        # x, y, std_dev, data_label = data[group_name]
        x = range(0, len(y_values[group_name]))
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
    ax.set_xlim(0, 30+end_extension)
    # ax.set_xticks(range(0, 300, 60))
    # ax.set_xticklabels([str(size // 60) for size in range(0, 300, 60)])
    ax.set_ylim(0, bandwidth_y_max)
    # ax.set_yticks(range(1, 11))
    # Set y-ticks at 0, 250, 500, 750, and 1000
    ax.set_yticks([0, 250, 500, 750, 1000])
    ax.set_yticklabels(['0', '250', '500', '750', '1,000'])

    # # Add vertical lines at experiment start and end
    # plt.axvline(x=vdb_start_offset_in_trace,
    #             color='blue',
    #             linestyle='--',
    #             linewidth=3)  #, label='Query Start')
    # plt.axvline(x=vdb_end_offset_in_trace,
    #             color='black',
    #             linestyle='--',
    #             linewidth=3)  #, label='Query End')

    plt.axvline(x=30, color='blue', linestyle='--', linewidth=3)
    if legend_label != None:
        # plt.legend(loc='lower right', fontsize=legend_font_size, labelspacing=0.1)
        plt.legend(loc='upper left',
                   fontsize=legend_font_size,
                   labelspacing=0.1,)

    plt.savefig(fig_save_path, bbox_inches='tight')
    plt.close()


PLOT_OPENAI_500K = True
if PLOT_OPENAI_500K:
    average_bandwidth['openai-500k'] = {}
    average_bandwidth_per_req['openai-500k'] = {}
    fig_save_path = os.path.join(fig_save_path_all,
                                 'fig-5-a-rw-bandwidth-openai-500k.pdf')
    # group_list = ['read', 'write']
    # y_values = {
    #     'read': io_size_by_sec['ALL_READS']['size'],
    #     'write': io_size_by_sec['ALL_WRITES']['size'],
    # }

    # currently we only plot read bandwidth
    group_list = bandwidth_trace['openai-500k'].keys()
    y_values = {}
    for cur_group in group_list:
        cur_start, cur_end = start_and_end_trace['openai-500k'][cur_group]
        cur_start = int(cur_start)
        cur_end = int(cur_end) + end_extension
        y_values[cur_group] = bandwidth_trace['openai-500k'][cur_group]['ALL_READS']['size'][cur_start: cur_end+1]
        cur_average_bandwidth = sum(y_values[cur_group]) / len(
            y_values[cur_group])
        average_bandwidth['openai-500k'][cur_group] = cur_average_bandwidth
        average_bandwidth_per_req['openai-500k'][
            cur_group] = cur_average_bandwidth / (
                qps['openai-500k'][cur_group] * 1000)
    std_dev = None
    # x_ticks = ['xtick_1', 'xtick_1']
    legend_label = {
        'milvus-diskann-con-1': '1 thread',
        'milvus-diskann-con-16': '16 threads',
        'milvus-diskann-con-256': '256 threads',
    }

    title = None
    xlabel = 'Time (S)'
    ylabel = 'Bandwidth (MiB/s)'

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

    for (index, group_name) in zip(range(len(group_list)), group_list):
        # Ensuring timing is equal for both lines
        io_size_by_sec = bandwidth_trace['openai-500k'][group_name]
        time_min = min(io_size_by_sec['ALL_READS']['ts'][0],
                       io_size_by_sec['ALL_WRITES']['ts'][0])
        time_max = max(io_size_by_sec['ALL_READS']['ts'][-1],
                       io_size_by_sec['ALL_WRITES']['ts'][-1])
        ran = time_max - time_min + 1

        # print(io_size_by_sec['ALL_READS'])
        # print(y_values[group_name])

        #### Currently, we only print read, so no need to show the write part
        # if (io_size_by_sec['ALL_READS']['ts'][0] > time_min):
        #     y_values['read'] = [0] * (io_size_by_sec['ALL_READS']['ts'][0] -
        #                             time_min) + y_values['read']
        # if (io_size_by_sec['ALL_READS']['ts'][-1] < time_max):
        #     y_values['read'] = y_values['read'] + [0] * (
        #         time_max - io_size_by_sec['ALL_READS']['ts'][-1])

        # if (io_size_by_sec['ALL_WRITES']['ts'][0] > time_min):
        #     y_values['write'] = [0] * (io_size_by_sec['ALL_WRITES']['ts'][0] -
        #                             time_min) + y_values['write']
        # if (io_size_by_sec['ALL_WRITES']['ts'][-1] < time_max):
        #     y_values['write'] = y_values['write'] + [0] * (
        #         time_max - io_size_by_sec['ALL_WRITES']['ts'][-1])

        # x, y, std_dev, data_label = data[group_name]
        x = range(0, len(y_values[group_name]))
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
    ax.set_xlim(0, 30+end_extension)
    # ax.set_xticks(range(0, 300, 60))
    # ax.set_xticklabels([str(size // 60) for size in range(0, 300, 60)])
    ax.set_ylim(0, bandwidth_y_max)
    # ax.set_yticks(range(1, 11))
    # Set y-ticks at 0, 250, 500, 750, and 1000
    ax.set_yticks([0, 250, 500, 750, 1000])
    ax.set_yticklabels(['0', '250', '500', '750', '1,000'])

    # # Add vertical lines at experiment start and end
    # plt.axvline(x=vdb_start_offset_in_trace,
    #             color='blue',
    #             linestyle='--',
    #             linewidth=3)  #, label='Query Start')
    # plt.axvline(x=vdb_end_offset_in_trace,
    #             color='black',
    #             linestyle='--',
    #             linewidth=3)  #, label='Query End')

    plt.axvline(x=30, color='blue', linestyle='--', linewidth=3)
    if legend_label != None:
        # plt.legend(loc='lower right', fontsize=legend_font_size, labelspacing=0.1)
        plt.legend(
            loc='upper left',
            fontsize=legend_font_size,
            labelspacing=0.1,
        )

    plt.savefig(fig_save_path, bbox_inches='tight')
    plt.close()


PLOT_OPENAI_5M = True
if PLOT_OPENAI_5M:
    average_bandwidth['openai-5m'] = {}
    average_bandwidth_per_req['openai-5m'] = {}
    fig_save_path = os.path.join(fig_save_path_all,
                                 'fig-5-a-rw-bandwidth-openai-5m.pdf')
    # group_list = ['read', 'write']
    # y_values = {
    #     'read': io_size_by_sec['ALL_READS']['size'],
    #     'write': io_size_by_sec['ALL_WRITES']['size'],
    # }

    # currently we only plot read bandwidth
    group_list = bandwidth_trace['openai-5m'].keys()
    y_values = {}
    for cur_group in group_list:
        cur_start, cur_end = start_and_end_trace['openai-5m'][cur_group]
        cur_start = int(cur_start)
        cur_end = int(cur_end) + end_extension
        y_values[cur_group] = bandwidth_trace['openai-5m'][cur_group]['ALL_READS']['size'][cur_start: cur_end+1]
        cur_average_bandwidth = sum(y_values[cur_group]) / len(
            y_values[cur_group])
        average_bandwidth['openai-5m'][cur_group] = cur_average_bandwidth
        average_bandwidth_per_req['openai-5m'][
            cur_group] = cur_average_bandwidth / (
                qps['openai-5m'][cur_group] * 1000)
    std_dev = None
    # x_ticks = ['xtick_1', 'xtick_1']
    legend_label = {
        'milvus-diskann-con-1': '1 thread',
        'milvus-diskann-con-4': '4 threads',
        'milvus-diskann-con-256': '256 threads',
    }

    title = None
    xlabel = 'Time (S)'
    ylabel = 'Bandwidth (MiB/s)'

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

    for (index, group_name) in zip(range(len(group_list)), group_list):
        # Ensuring timing is equal for both lines
        io_size_by_sec = bandwidth_trace['openai-5m'][group_name]
        time_min = min(io_size_by_sec['ALL_READS']['ts'][0],
                       io_size_by_sec['ALL_WRITES']['ts'][0])
        time_max = max(io_size_by_sec['ALL_READS']['ts'][-1],
                       io_size_by_sec['ALL_WRITES']['ts'][-1])
        ran = time_max - time_min + 1

        # print(io_size_by_sec['ALL_READS'])
        # print(y_values[group_name])

        #### Currently, we only print read, so no need to show the write part
        # if (io_size_by_sec['ALL_READS']['ts'][0] > time_min):
        #     y_values['read'] = [0] * (io_size_by_sec['ALL_READS']['ts'][0] -
        #                             time_min) + y_values['read']
        # if (io_size_by_sec['ALL_READS']['ts'][-1] < time_max):
        #     y_values['read'] = y_values['read'] + [0] * (
        #         time_max - io_size_by_sec['ALL_READS']['ts'][-1])

        # if (io_size_by_sec['ALL_WRITES']['ts'][0] > time_min):
        #     y_values['write'] = [0] * (io_size_by_sec['ALL_WRITES']['ts'][0] -
        #                             time_min) + y_values['write']
        # if (io_size_by_sec['ALL_WRITES']['ts'][-1] < time_max):
        #     y_values['write'] = y_values['write'] + [0] * (
        #         time_max - io_size_by_sec['ALL_WRITES']['ts'][-1])

        # x, y, std_dev, data_label = data[group_name]
        x = range(0, len(y_values[group_name]))
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
    ax.set_xlim(0, 30+end_extension)
    # ax.set_xticks(range(0, 300, 60))
    # ax.set_xticklabels([str(size // 60) for size in range(0, 300, 60)])
    ax.set_ylim(0, bandwidth_y_max)
    # ax.set_yticks(range(1, 11))
    # Set y-ticks at 0, 250, 500, 750, and 1000
    ax.set_yticks([0, 250, 500, 750, 1000])
    ax.set_yticklabels(['0', '250', '500', '750', '1,000'])

    # # Add vertical lines at experiment start and end
    # plt.axvline(x=vdb_start_offset_in_trace,
    #             color='blue',
    #             linestyle='--',
    #             linewidth=3)  #, label='Query Start')
    # plt.axvline(x=vdb_end_offset_in_trace,
    #             color='black',
    #             linestyle='--',
    #             linewidth=3)  #, label='Query End')

    plt.axvline(x=30, color='blue', linestyle='--', linewidth=3)
    if legend_label != None:
        # plt.legend(loc='lower right', fontsize=legend_font_size, labelspacing=0.1)
        plt.legend(loc='upper left',
                   fontsize=legend_font_size,
                   labelspacing=0.1,)

    plt.savefig(fig_save_path, bbox_inches='tight')
    plt.close()


# print the average value
print('Average bandwidth:')
for dataset, values in average_bandwidth.items():
    print(f"{dataset}:")
    for group, avg_bandwidth in values.items():
        print(f"  {group}: {(avg_bandwidth/1024):.2f} MiB/s")

print('Average bandwidth per request:')
for dataset, values in average_bandwidth_per_req.items():
    print(f"{dataset}:")
    for group, avg_bandwidth in values.items():
        print(f"  {group}: {(avg_bandwidth/1024):.2f} MiB/s")

# Now let's plot the average bandwidth, just plot 2 figures

average_bandwidth_group_list = []
for cur_db in vdb_configs:
    for cur_index in vdb_configs[cur_db]:
        average_bandwidth_group_list.append(f"{cur_db}-{cur_index}")

# average bandwidth: CON = 1
if True:
    # Data, set unused value to none
    fig_save_path = os.path.join(fig_save_path_all,
                                 'fig-6-a-avg-bandwidth-con-1.pdf')
    group_list = average_bandwidth_group_list
    y_values = {}
    for cur_vdb_setup in group_list:
        y_values[cur_vdb_setup] = []
        for cur_dataset in datasets:
            y_values[cur_vdb_setup].append(average_bandwidth_per_req[cur_dataset][f'{cur_vdb_setup}-con-1'])
    std_dev = None
    x_ticks = datasets
    legend_label = {g:g for g in group_list}

    title = None
    xlabel = None
    ylabel = 'Bandwidth (MiB/s)'

    # Parameters
    bar_width = 0.2

    datalabel_size = 26
    datalabel_va = 'bottom'
    axis_tick_font_size = 34
    axis_label_font_size = 44
    legend_font_size = 30

    # plot
    reset_color()
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.grid(axis='y')  # x, y, both

    # x, y axis limit
    # ax.set_xlim(0, 2)
    ax.set_ylim(0, 2.5)
    # Rotate x-tick labels by 60 degrees
    plt.xticks(list(np.arange(len(x_ticks))), x_ticks, rotation=30)

    if title:
        plt.title(title)

    if xlabel:
        plt.xlabel(xlabel, fontsize=axis_label_font_size)
    if ylabel:
        plt.ylabel(ylabel, fontsize=axis_label_font_size)

    ax.tick_params(axis='both', which='major', labelsize=axis_tick_font_size)

    # compute bar offset, with respect to center
    bar_offset = []
    mid_point = (len(group_list) * bar_width) / 2
    for i in range(len(group_list)):
        bar_offset.append(bar_width * i + 0.5 * bar_width - mid_point)

    x_axis = np.arange(len(x_ticks))
    # draw figure by column
    for (index, group_name) in zip(range(len(group_list)), group_list):
        y = y_values[group_name]
        y = [i/1024 for i in y]
        yerr = None
        if std_dev:
            yerr = std_dev[group_name]
        bar_pos = x_axis + bar_offset[index]

        plt.bar(bar_pos,
                y,
                width=bar_width,
                label=legend_label[group_name],
                yerr=yerr,
                color=get_next_color())

        # print data label
        for (x, y) in zip(bar_pos, y):
            text = '{:.2f}'.format(y)
            plt.text(
                x,
                y,
                text,
                size=datalabel_size,
                ha='center',
                va=
                datalabel_va,  # 'bottom', 'baseline', 'center', 'center_baseline', 'top'
            )

    # Legend: Change the ncol and loc to fine-tune the location of legend
    # if legend_label != None:
    #     plt.legend(fontsize=legend_font_size)
    #     # plt.legend(fontsize=legend_font_size,
    #     #            ncol=2,å
    #     #            loc='upper left',
    #     #            bbox_to_anchor=(0, 1.2),
    #     #            columnspacing=0.3)

    plt.savefig(fig_save_path, bbox_inches='tight')
    plt.close()


# Avaerage bandwidth, CON=256
if True:
    # Data, set unused value to none
    fig_save_path = os.path.join(fig_save_path_all,
                                 'fig-6-b-avg-bandwidth-con-256.pdf')
    group_list = average_bandwidth_group_list
    y_values = {}
    for cur_vdb_setup in group_list:
        y_values[cur_vdb_setup] = []
        for cur_dataset in datasets:
            y_values[cur_vdb_setup].append(
                average_bandwidth_per_req[cur_dataset]
                [f'{cur_vdb_setup}-con-256'])
    std_dev = None
    x_ticks = datasets
    legend_label = {g: g for g in group_list}

    title = None
    xlabel = None
    ylabel = 'Bandwidth (MiB/s)'

    # Parameters
    bar_width = 0.2

    datalabel_size = 26
    datalabel_va = 'bottom'
    axis_tick_font_size = 34
    axis_label_font_size = 44
    legend_font_size = 30

    # plot
    reset_color()
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.grid(axis='y')  # x, y, both

    # x, y axis limit
    # ax.set_xlim(0, 2)
    ax.set_ylim(0, 2.5)
    # Rotate x-tick labels by 60 degrees
    plt.xticks(list(np.arange(len(x_ticks))), x_ticks, rotation=30)

    if title:
        plt.title(title)

    if xlabel:
        plt.xlabel(xlabel, fontsize=axis_label_font_size)
    if ylabel:
        plt.ylabel(ylabel, fontsize=axis_label_font_size)

    ax.tick_params(axis='both', which='major', labelsize=axis_tick_font_size)

    # compute bar offset, with respect to center
    bar_offset = []
    mid_point = (len(group_list) * bar_width) / 2
    for i in range(len(group_list)):
        bar_offset.append(bar_width * i + 0.5 * bar_width - mid_point)

    x_axis = np.arange(len(x_ticks))
    # draw figure by column
    for (index, group_name) in zip(range(len(group_list)), group_list):
        y = y_values[group_name]
        y = [x/1024 for x in y]
        yerr = None
        if std_dev:
            yerr = std_dev[group_name]
        bar_pos = x_axis + bar_offset[index]

        plt.bar(bar_pos,
                y,
                width=bar_width,
                label=legend_label[group_name],
                yerr=yerr,
                color=get_next_color())

        # print data label
        for (x, y) in zip(bar_pos, y):
            text = '{:.2f}'.format(y)
            plt.text(
                x,
                y,
                text,
                size=datalabel_size,
                ha='center',
                va=
                datalabel_va,  # 'bottom', 'baseline', 'center', 'center_baseline', 'top'
            )

    # Legend: Change the ncol and loc to fine-tune the location of legend
    # if legend_label != None:
    #     plt.legend(fontsize=legend_font_size)
    #     # plt.legend(fontsize=legend_font_size,
    #     #            ncol=2,å
    #     #            loc='upper left',
    #     #            bbox_to_anchor=(0, 1.2),
    #     #            columnspacing=0.3)

    plt.savefig(fig_save_path, bbox_inches='tight')
    plt.close()
