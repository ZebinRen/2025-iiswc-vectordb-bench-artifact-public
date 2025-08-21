# Plot the latency and throughput of all databases
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

vector_db_output_dir = 'results-performance'
figure_dir = 'figures'
os.makedirs(figure_dir, exist_ok=True)


# Parse results
results_qps = {}
results_latency = {}
results_latency_avg = {}
results_latency_p99 = {}
results_recall = {}
results_qps_std = {}
results_latency_std = {}
results_latency_avg_std = {}
results_latency_p99_std = {}
results_recall_std = {}
plots_groups = []

for cur_dataset in datasets:
    results_qps[cur_dataset] = {}
    results_latency[cur_dataset] = {}
    results_latency_avg[cur_dataset] = {}
    results_latency_p99[cur_dataset] = {}
    results_recall[cur_dataset] = {}
    results_qps_std[cur_dataset] = {}
    results_latency_std[cur_dataset] = {}
    results_latency_avg_std[cur_dataset] = {}
    results_latency_p99_std[cur_dataset] = {}
    results_recall_std[cur_dataset] = {}
    for cur_vdb in vector_dbs:
        for cur_index in vdb_configs[cur_vdb]:
            cur_db_index = f"{cur_vdb}-{cur_index}"
            results_qps[cur_dataset][cur_db_index] = []
            results_latency[cur_dataset][cur_db_index] = []
            results_latency_avg[cur_dataset][cur_db_index] = []
            results_latency_p99[cur_dataset][cur_db_index] = []
            results_qps_std[cur_dataset][cur_db_index] = []
            results_latency_std[cur_dataset][cur_db_index] = []
            results_latency_avg_std[cur_dataset][cur_db_index] = []
            results_latency_p99_std[cur_dataset][cur_db_index] = []
            for cur_concurrency in all_concurrency:
                cur_rep_results_qps = []
                cur_rep_results_latency = []
                cur_rep_results_latency_avg = []
                cur_rep_results_latency_p99 = []
                cur_rep_results_recall = []
                for cur_rep in range(rep):
                    vdb_output_file = f'vdb-bench-{cur_dataset}-con-{cur_concurrency}-{expr_config_fname[cur_vdb][cur_index+"-"+cur_dataset]}-rep-{cur_rep}.log'

                    vdb_bench_output_path = os.path.join(
                        vector_db_output_dir,
                        f'{cur_vdb}-{cur_index}-{cur_dataset}',
                        f'{cur_vdb}-{cur_index}-{cur_dataset}-output',
                        vdb_output_file)

                    # Check if the file exists before parsing
                    if not os.path.exists(vdb_bench_output_path):
                        # print('File not found:', vdb_bench_output_path)
                        continue

                    vectordb_perf = parse.parse_vectordb_bench_output(
                        vdb_bench_output_path)
                    cur_rep_results_qps.append(vectordb_perf['qps'] / 1000) # QPS at the unit of K
                    cur_rep_results_latency.append(vectordb_perf['latency'])
                    cur_rep_results_recall.append(vectordb_perf['recall'])
                    cur_rep_results_latency_avg.append(vectordb_perf['con_latency_avg'])
                    cur_rep_results_latency_p99.append(vectordb_perf['con_latency_p99'])

                if len(cur_rep_results_qps) == 0:
                    continue
                results_qps[cur_dataset][cur_db_index].append(sum(cur_rep_results_qps) / len(cur_rep_results_qps))
                results_latency[cur_dataset][cur_db_index].append(sum(cur_rep_results_latency) / len(cur_rep_results_latency))
                results_latency_avg[cur_dataset][cur_db_index].append(sum(cur_rep_results_latency_avg) / len(cur_rep_results_latency_avg))
                results_latency_p99[cur_dataset][cur_db_index].append(sum(cur_rep_results_latency_p99) / len(cur_rep_results_latency_p99))
                results_recall[cur_dataset][cur_db_index] = sum(cur_rep_results_recall) / len(cur_rep_results_recall)
                results_qps_std[cur_dataset][cur_db_index].append(float(np.std(cur_rep_results_qps)))
                results_latency_std[cur_dataset][cur_db_index].append(float(np.std(cur_rep_results_latency)))
                results_latency_avg_std[cur_dataset][cur_db_index].append(float(np.std(cur_rep_results_latency_avg)))
                results_latency_p99_std[cur_dataset][cur_db_index].append(float(np.std(cur_rep_results_latency_p99)))
                results_recall_std[cur_dataset][cur_db_index] = float(np.std(cur_rep_results_recall))

# Print the exact values for computation
for cur_dataset in datasets:
    print(f"QPS Dataset: {cur_dataset}")
    for cur_vdb in vector_dbs:
        for cur_index in vdb_configs[cur_vdb]:
            cur_db_index = f"{cur_vdb}-{cur_index}"
            printed_qps = [
                f'{x:.2f}' for x in results_qps[cur_dataset][cur_db_index]
            ]
            print(f"  {cur_db_index} QPS: {printed_qps}")

for cur_dataset in datasets:
    print(f"Latency AVG Dataset: {cur_dataset}")
    for cur_vdb in vector_dbs:
        for cur_index in vdb_configs[cur_vdb]:
            cur_db_index = f"{cur_vdb}-{cur_index}"
            printed_lat_avg = [
                f'{x:.2f}' for x in results_latency_avg[cur_dataset][cur_db_index]
            ]
            print(
                f"  {cur_db_index} Latency Avg: {printed_lat_avg}"
            )

for cur_dataset in datasets:
    print(f"Latency P99 Dataset: {cur_dataset}")
    for cur_vdb in vector_dbs:
        for cur_index in vdb_configs[cur_vdb]:
            cur_db_index = f"{cur_vdb}-{cur_index}"
            printed_lat_p99 = [
                f'{x:.2f}' for x in results_latency_p99[cur_dataset][cur_db_index]
            ]
            print(
                f"  {cur_db_index} Latency P99: {printed_lat_p99}"
            )


from plot import *

LAT_AVG_SCALE = 'log'
LAT_P99_SCALE = 'log'
MIN_Y_LAT_AVG = 1
MAX_Y_LAT_AVG = 10000
MIN_Y_LAT_P99 = 1
MAX_Y_LAT_P99 = 10000

legend_label = {
    'milvus-ivf': 'Milvus IVF',
    'milvus-hnsw': 'Milvus HNSW',
    'milvus-diskann': 'Milvus DiskANN',
    'qdrant-hnsw-mem': 'Qdrant HNSW',
    'weaviate-hnsw': 'Weaviate HNSW',
    'lancedb-embedded-ivf': 'LanceDB IVFPQ',
    'lancedb-embedded-hnsw': 'LanceDB HNSW',
}

DRAW_COHERE_1M = True
if DRAW_COHERE_1M:
    # Cohere 1M QPS
    if True:
        # QPS
        fig_save_path = os.path.join(figure_dir, 'fig-2-a-cohere-1m-qps.pdf')
        group_list = []
        for cur_db in vdb_configs:
            for index in vdb_configs[cur_db]:
                group_list.append(f"{cur_db}-{index}")
        y_values = results_qps['cohere-1m']
        std_dev = results_qps_std['cohere-1m']
        x_ticks = [str(x) for x in all_concurrency]
        # legend_label = {i: i for i in group_list}
        # print(f"QPS cohere 1M: {y_values}")


        title = None
        xlabel = 'Concurrency'
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

        ax.tick_params(axis='both', which='major', labelsize=axis_tick_font_size)
        ax.xaxis.set_ticks(range(1, len(x_ticks) + 1))
        ax.set_xticklabels(x_ticks)
        ax.set_xlim(0, 9.5)
        ax.set_ylim(0, 8)

        for (index, group_name) in zip(range(len(group_list)), group_list):
            # x, y, std_dev, data_label = data[group_name]
            x = range(1, len(y_values[group_name]) + 1)
            y = y_values[group_name]
            yerr = None
            if std_dev:
                yerr = std_dev[group_name]

            linestyle = 'dashed' if group_name in vdb_disk else 'solid'
            plt.errorbar(
                x,
                y,
                yerr=yerr,
                label=legend_label[group_name],
                marker=dot_style[index % len(dot_style)],
                linewidth=linewidth,
                markersize=markersize,
                linestyle=linestyle,
                color=get_next_color(),
            )
            # Add data label
            # for i in range(len(data_label)):
            #     ax.text(x[i], y[i], data_label[i], size=datalabel_size)

        if legend_label != None:
            plt.legend(fontsize=legend_font_size, labelspacing=0.1, ncol=1, columnspacing=0.1)
            # plt.legend(fontsize=legend_font_size,
            #            ncol=2,
            #            loc='upper left',
            #            bbox_to_anchor=(0, 1.2),
            #            columnspacing=0.3)

        plt.savefig(fig_save_path, bbox_inches='tight')
        plt.close()

    # Cohere 1M Latency P99
    if True:
        fig_save_path = os.path.join(figure_dir, 'fig-3-a-cohere-1m-latency-p99.pdf')
        group_list = []
        for cur_db in vdb_configs:
            for index in vdb_configs[cur_db]:
                group_list.append(f"{cur_db}-{index}")
        y_values = results_latency_p99['cohere-1m']
        std_dev = results_latency_p99_std['cohere-1m']
        x_ticks = [str(x) for x in all_concurrency]
        # legend_label = {i: i for i in group_list}

        title = None
        xlabel = 'Concurrency'
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
        if LAT_P99_SCALE == 'log':
            ax.set_yscale('log')

        plt.xlabel(xlabel, fontsize=axis_label_font_size)
        plt.ylabel(ylabel, fontsize=axis_label_font_size)
        plt.grid(True)

        ax.tick_params(axis='both',
                       which='major',
                       labelsize=axis_tick_font_size)
        ax.xaxis.set_ticks(range(1, len(x_ticks) + 1))
        ax.set_xticklabels(x_ticks)
        ax.set_xlim(0, 9.5)
        ax.set_ylim(MIN_Y_LAT_P99, MAX_Y_LAT_P99)

        for (index, group_name) in zip(range(len(group_list)), group_list):
            # x, y, std_dev, data_label = data[group_name]
            x = range(1, len(y_values[group_name]) + 1)
            y = y_values[group_name]
            yerr = None
            if std_dev:
                yerr = std_dev[group_name]
            linestyle = 'dashed' if group_name in vdb_disk else 'solid'

            plt.errorbar(
                x,
                y,
                yerr=yerr,
                label=legend_label[group_name],
                marker=dot_style[index % len(dot_style)],
                linewidth=linewidth,
                markersize=markersize,
                linestyle=linestyle,
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


DRAW_COHERE_10M = True
if DRAW_COHERE_10M:
    # Cohere 10M QPS
    if True:
        fig_save_path = os.path.join(figure_dir, 'fig-2-b-cohere-10m-qps.pdf')
        group_list = []
        for cur_db in vdb_configs:
            for index in vdb_configs[cur_db]:
                group_list.append(f"{cur_db}-{index}")
        y_values = results_qps['cohere-10m']
        std_dev = results_qps_std['cohere-10m']
        x_ticks = [str(x) for x in all_concurrency]
        # legend_label = {i: i for i in group_list}
        # print(f"QPS cohere 10M: {y_values}")

        title = None
        xlabel = 'Concurrency'
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

        ax.tick_params(axis='both', which='major', labelsize=axis_tick_font_size)
        ax.xaxis.set_ticks(range(1, len(x_ticks) + 1))
        ax.set_xticklabels(x_ticks)
        ax.set_xlim(0, 9.5)
        ax.set_ylim(0, 2)

        for (index, group_name) in zip(range(len(group_list)), group_list):
            # x, y, std_dev, data_label = data[group_name]
            x = range(1, len(y_values[group_name]) + 1)
            y = y_values[group_name]
            yerr = None
            if std_dev:
                yerr = std_dev[group_name]
            linestyle = 'dashed' if group_name in vdb_disk else 'solid'

            plt.errorbar(
                x,
                y,
                yerr=yerr,
                label=legend_label[group_name],
                marker=dot_style[index % len(dot_style)],
                linewidth=linewidth,
                markersize=markersize,
                linestyle=linestyle,
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

    # Cohere 10M Latency P99
    if True:
        fig_save_path = os.path.join(figure_dir, 'fig-3-b-cohere-10m-latency-p99.pdf')
        group_list = []
        for cur_db in vdb_configs:
            for index in vdb_configs[cur_db]:
                group_list.append(f"{cur_db}-{index}")
        y_values = results_latency_p99['cohere-10m']
        std_dev = results_latency_p99_std['cohere-10m']
        x_ticks = [str(x) for x in all_concurrency]
        # legend_label = {i: i for i in group_list}

        title = None
        xlabel = 'Concurrency'
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
        if LAT_P99_SCALE == 'log':
            ax.set_yscale('log')

        plt.xlabel(xlabel, fontsize=axis_label_font_size)
        plt.ylabel(ylabel, fontsize=axis_label_font_size)
        plt.grid(True)

        ax.tick_params(axis='both', which='major', labelsize=axis_tick_font_size)
        ax.xaxis.set_ticks(range(1, len(x_ticks) + 1))
        ax.set_xticklabels(x_ticks)
        ax.set_xlim(0, 9.5)
        ax.set_ylim(MIN_Y_LAT_P99, MAX_Y_LAT_P99)

        for (index, group_name) in zip(range(len(group_list)), group_list):
            # x, y, std_dev, data_label = data[group_name]
            x = range(1, len(y_values[group_name]) + 1)
            y = y_values[group_name]
            yerr = None
            if std_dev:
                yerr = std_dev[group_name]
            linestyle = 'dashed' if group_name in vdb_disk else 'solid'

            plt.errorbar(
                x,
                y,
                yerr=yerr,
                label=legend_label[group_name],
                marker=dot_style[index % len(dot_style)],
                linewidth=linewidth,
                markersize=markersize,
                linestyle=linestyle,
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

DRAW_OPENAI_500K = True
if DRAW_OPENAI_500K:
    # Openai 500K QPS
    if True:
        fig_save_path = os.path.join(figure_dir, 'fig-2-c-openai-500k-qps.pdf')
        group_list = []
        for cur_db in vdb_configs:
            for index in vdb_configs[cur_db]:
                group_list.append(f"{cur_db}-{index}")
        y_values = results_qps['openai-500k']
        std_dev = results_qps_std['openai-500k']
        x_ticks = [str(x) for x in all_concurrency]
        # legend_label = {i: i for i in group_list}
        # print(f"QPS openai 500K: {y_values}")

        title = None
        xlabel = 'Concurrency'
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

        ax.tick_params(axis='both', which='major', labelsize=axis_tick_font_size)
        ax.xaxis.set_ticks(range(1, len(x_ticks) + 1))
        ax.set_xticklabels(x_ticks)
        # ax.set_xlim(0, 8)
        ax.set_ylim(0, 8)

        for (index, group_name) in zip(range(len(group_list)), group_list):
            # x, y, std_dev, data_label = data[group_name]
            x = range(1, len(y_values[group_name]) + 1)
            y = y_values[group_name]
            yerr = None
            if std_dev:
                yerr = std_dev[group_name]
            linestyle = 'dashed' if group_name in vdb_disk else 'solid'

            plt.errorbar(
                x,
                y,
                yerr=yerr,
                label=legend_label[group_name],
                marker=dot_style[index % len(dot_style)],
                linewidth=linewidth,
                markersize=markersize,
                linestyle=linestyle,
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

    # Openai 500k Latency P99
    if True:
        fig_save_path = os.path.join(figure_dir, 'fig-3-c-openai-500k-latency-p99.pdf')
        group_list = []
        for cur_db in vdb_configs:
            for index in vdb_configs[cur_db]:
                group_list.append(f"{cur_db}-{index}")
        y_values = results_latency_p99['openai-500k']
        std_dev = results_latency_p99_std['openai-500k']
        x_ticks = [str(x) for x in all_concurrency]
        # legend_label = {i: i for i in group_list}

        title = None
        xlabel = 'Concurrency'
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
        if LAT_P99_SCALE == 'log':
            ax.set_yscale('log')

        plt.xlabel(xlabel, fontsize=axis_label_font_size)
        plt.ylabel(ylabel, fontsize=axis_label_font_size)
        plt.grid(True)

        ax.tick_params(axis='both',
                       which='major',
                       labelsize=axis_tick_font_size)
        ax.xaxis.set_ticks(range(1, len(x_ticks) + 1))
        ax.set_xticklabels(x_ticks)
        ax.set_xlim(0, 9.5)
        ax.set_ylim(MIN_Y_LAT_P99, MAX_Y_LAT_P99)

        for (index, group_name) in zip(range(len(group_list)), group_list):
            # x, y, std_dev, data_label = data[group_name]
            x = range(1, len(y_values[group_name]) + 1)
            y = y_values[group_name]
            yerr = None
            if std_dev:
                yerr = std_dev[group_name]
            linestyle = 'dashed' if group_name in vdb_disk else 'solid'

            plt.errorbar(
                x,
                y,
                yerr=yerr,
                label=legend_label[group_name],
                marker=dot_style[index % len(dot_style)],
                linewidth=linewidth,
                markersize=markersize,
                linestyle=linestyle,
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

DRAW_OPENAI_5M = True
if DRAW_OPENAI_5M:
    # Openai 5M QPS
    if True:
        fig_save_path = os.path.join(figure_dir, 'fig-2-d-openai-5m-qps.pdf')
        group_list = []
        for cur_db in vdb_configs:
            for index in vdb_configs[cur_db]:
                group_list.append(f"{cur_db}-{index}")
        y_values = results_qps['openai-5m']
        std_dev = results_qps_std['openai-5m']
        x_ticks = [str(x) for x in all_concurrency]
        # legend_label = {i: i for i in group_list}
        # print(f"QPS openai 5M: {y_values}")

        title = None
        xlabel = 'Concurrency'
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

        ax.tick_params(axis='both', which='major', labelsize=axis_tick_font_size)
        ax.xaxis.set_ticks(range(1, len(x_ticks) + 1))
        ax.set_xticklabels(x_ticks)
        ax.set_xlim(0, 9.5)
        ax.set_ylim(0, 2)

        for (index, group_name) in zip(range(len(group_list)), group_list):
            # x, y, std_dev, data_label = data[group_name]
            x = range(1, len(y_values[group_name]) + 1)
            y = y_values[group_name]
            yerr = None
            if std_dev:
                yerr = std_dev[group_name]
            linestyle = 'dashed' if group_name in vdb_disk else 'solid'

            plt.errorbar(
                x,
                y,
                yerr=yerr,
                label=legend_label[group_name],
                marker=dot_style[index % len(dot_style)],
                linewidth=linewidth,
                markersize=markersize,
                linestyle=linestyle,
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

    # Openai 5M Latency P99
    if True:
        fig_save_path = os.path.join(figure_dir, 'fig-3-d-openai-5m-latency-p99.pdf')
        group_list = []
        for cur_db in vdb_configs:
            for index in vdb_configs[cur_db]:
                group_list.append(f"{cur_db}-{index}")
        y_values = results_latency_p99['openai-5m']
        std_dev = results_latency_p99_std['openai-5m']
        x_ticks = [str(x) for x in all_concurrency]
        # legend_label = {i: i for i in group_list}

        title = None
        xlabel = 'Concurrency'
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
        if LAT_P99_SCALE == 'log':
            ax.set_yscale('log')

        plt.xlabel(xlabel, fontsize=axis_label_font_size)
        plt.ylabel(ylabel, fontsize=axis_label_font_size)
        plt.grid(True)

        ax.tick_params(axis='both',
                       which='major',
                       labelsize=axis_tick_font_size)
        ax.xaxis.set_ticks(range(1, len(x_ticks) + 1))
        ax.set_xticklabels(x_ticks)
        ax.set_xlim(0, 9.5)
        ax.set_ylim(MIN_Y_LAT_P99, MAX_Y_LAT_P99)

        for (index, group_name) in zip(range(len(group_list)), group_list):
            # x, y, std_dev, data_label = data[group_name]
            x = range(1, len(y_values[group_name]) + 1)
            y = y_values[group_name]
            yerr = None
            if std_dev:
                yerr = std_dev[group_name]
            linestyle = 'dashed' if group_name in vdb_disk else 'solid'

            plt.errorbar(
                x,
                y,
                yerr=yerr,
                label=legend_label[group_name],
                marker=dot_style[index % len(dot_style)],
                linewidth=linewidth,
                markersize=markersize,
                linestyle=linestyle,
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
