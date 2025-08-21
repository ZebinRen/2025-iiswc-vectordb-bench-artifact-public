import re
from datetime import datetime, timedelta


def parse_vectordb_bench_output(log_path):
    """
    Parses a log file for the line containing Metric(...) and returns
    a dict with qps, latency (serial_latency_p99), recall,
    conc_latency_p99_list, and conc_latency_avg_list as floats.
    Latencies are returned in milliseconds.

    Parameters
    ----------
    log_path : str
        Path to the log file.

    Returns
    -------
    dict or None
        {'qps': float,
         'latency': float,
         'recall': float,
         'conc_latency_p99_list': List[float],
         'conc_latency_avg_list': List[float]}
        Returns None if required metrics are missing.
    """
    base_pattern = re.compile(
        r"qps=(?P<qps>[\d\.]+).*?"
        r"serial_latency_p99=(?:np\.float64\()?(?P<latency>[\d\.]+)\)?.*?"
        r"recall=(?:np\.float64\()?(?P<recall>[\d\.]+)\)?")
    conc_pattern = re.compile(r"conc_latency_p99_list=\[(?P<p99>[^\]]*)\].*?"
                              r"conc_latency_avg_list=\[(?P<avg>[^\]]*)\]")

    found_base = False
    found_conc = False

    with open(log_path, 'r') as f:
        for line in f:
            m = base_pattern.search(line)
            if not m:
                continue
            found_base = True

            qps = float(m.group('qps'))
            latency = float(
                m.group('latency')
            ) * 1000  # the metric is in second, which is the return value of time.perf_counter()
            recall = float(m.group('recall'))

            conc_latency_p99_list = []
            conc_latency_avg_list = []
            m2 = conc_pattern.search(line)
            if m2:
                found_conc = True
                p99_vals = [
                    float(val) for val in re.findall(
                        r"np\.float64\(([-+Ee\d\.]+)\)", m2.group('p99'))
                ]
                avg_vals = [
                    float(val) for val in re.findall(
                        r"np\.float64\(([-+Ee\d\.]+)\)", m2.group('avg'))
                ]
                conc_latency_p99_list = [v * 1000 for v in p99_vals]
                conc_latency_avg_list = [v * 1000 for v in avg_vals]

            result = {
                'qps': qps,
                'latency': latency,
                'recall': recall,
                'con_latency_p99': conc_latency_p99_list[0],
                'con_latency_avg': conc_latency_avg_list[0],
            }
            if not found_conc:
                print(
                    f"Warning: concurrent latency metrics not found in {log_path}"
                )
            return result

    # If we reach here, base metrics were not found
    missing = []
    if not found_base:
        missing.append('qps, serial_latency_p99, recall')
    # found_conc is False means conc lists missing, but we warn earlier
    if missing:
        print(f"Error: Missing metrics {', '.join(missing)} in {log_path}")
        raise ValueError("Required metrics not found in log file.")
    return None



def parse_experiment_times(
        log_path: str):
    """
    Reads the given log file and finds:
      - the time of the line containing "Syncing all process and start concurrency search"
      - the time of the line containing "End search in concurrency"

    Returns a tuple (start_time, end_time) where each is a string "HH:MM:SS",
    or None if the corresponding line wasn't found.
    """
    start_time = None
    end_time = None

    # Regex to capture the timestamp HH:MM:SS from lines like:
    # "2025-06-08 19:31:06,130 | INFO: Syncing all process..."
    time_re = re.compile(r'^\d{4}-\d{2}-\d{2} (\d{2}:\d{2}:\d{2}),\d+')

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            if start_time is None and "Syncing all process and start concurrency search" in line:
                m = time_re.match(line)
                if m:
                    start_time = m.group(1)
            elif end_time is None and "End search in concurrency" in line:
                m = time_re.match(line)
                if m:
                    end_time = m.group(1)
            # stop early if both found
            if start_time and end_time:
                break

    return start_time, end_time

def parse_cpu_trace(fpath):
    usage_by_class = {}
    times = []
    header_found = False
    classes_start_idx = None

    f = open(fpath, 'r')
    lines = f.readlines()
    f.close()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        tokens = line.split()
        if not header_found and 'CPU' in tokens and '%usr' in tokens:
            cpu_idx = tokens.index('CPU')
            classes = tokens[cpu_idx + 1:]
            usage_by_class = {cls_name: [] for cls_name in classes}
            classes_start_idx = cpu_idx + 1
            header_found = True
            continue

        if header_found:
            if len(tokens) < classes_start_idx + len(usage_by_class):
                continue

            # tokens[0] is e.g. "01:45:41", tokens[1] is "AM" or "PM"
            ts = f"{tokens[0]} {tokens[1]}"
            try:
                t24 = datetime.strptime(ts, '%I:%M:%S %p').strftime('%H:%M:%S')
            except ValueError:
                # if something unexpected, just keep the raw time
                t24 = tokens[0]
            times.append(t24)

            usage_values = tokens[classes_start_idx:]
            for cls_name, val_str in zip(usage_by_class.keys(), usage_values):
                try:
                    usage_by_class[cls_name].append(float(val_str))
                except ValueError:
                    pass

    old_usage_by_class = usage_by_class.copy()
    usage_by_class = {'time': times}
    for k, v in old_usage_by_class.items():
        usage_by_class[k[1:]] = v

    return usage_by_class


def find_time_indices(times, start_time, end_time):
    """
    times       : list of 'HH:MM:SS' strings, sorted ascending
    start_time  : 'HH:MM:SS' string marking beginning (inclusive)
    end_time    : 'HH:MM:SS' string marking end       (inclusive)
    
    Returns (start_idx, end_idx), using a ±1s fallback if needed.
    Raises ValueError if neither the exact nor the ±1s‐adjusted times are found,
    or if start ends up after end.
    """

    # helper to adjust a timestamp string by delta seconds
    def adjust(ts_str, delta):
        dt = datetime.strptime(
            ts_str, '%H:%M:%S') + timedelta(seconds=delta)
        return dt.strftime('%H:%M:%S')

    # 1) locate start
    if start_time in times:
        start_idx = times.index(start_time)
        actual_start = start_time
    else:
        alt_start = adjust(start_time, +1)
        if alt_start in times:
            start_idx = times.index(alt_start)
            actual_start = alt_start
        else:
            raise ValueError(
                f"Start time {start_time!r} not found, nor even {alt_start!r}")

    # 2) locate end
    if end_time in times:
        # last occurrence in case of duplicates
        end_idx = len(times) - 1 - times[::-1].index(end_time)
        actual_end = end_time
    else:
        alt_end = adjust(end_time, -1)
        if alt_end in times:
            end_idx = len(times) - 1 - times[::-1].index(alt_end)
            actual_end = alt_end
        else:
            raise ValueError(
                f"End time {end_time!r} not found, nor even {alt_end!r}")

    # sanity check
    if start_idx > end_idx:
        raise ValueError(
            f"After adjustment, start ({actual_start}) comes after end ({actual_end})."
        )

    return start_idx, end_idx

def clip_cpu_trace(cpu_traces, start, end):
    start_offset, end_offset = find_time_indices(cpu_traces['time'], start, end)
    clipped_traces = {k: v[start_offset:end_offset + 1] for k, v in cpu_traces.items()}

    return clipped_traces

def parse_mem_trace(fpath):
    # size in MiB
    fields = ['time', 'total', 'used', 'free', 'shared', 'buff/cache', 'available']
    usage_by_field = {field: [] for field in fields}

    # Read all lines at once
    f = open(fpath, 'r')
    lines = f.readlines()
    f.close()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        tokens = line.split()
        if len(tokens) < 8 or tokens[1] != 'Mem:':
            continue

        values = [tokens[0]] + tokens[2:8]
        if len(values) != len(fields):
            continue

        for field, val_str in zip(fields, values):
            if field == 'time':
                usage_by_field[field].append(val_str)
            else:
                try:
                    usage_by_field[field].append(int(val_str))
                except ValueError:
                    usage_by_field[field].append(None)

    return usage_by_field

def parse_page_cache_trace(fpath):
    fields = ['time', 'nr_file_pages']
    usage_by_field = {field: [] for field in fields}

    f = open(fpath, 'r')
    lines = f.readlines()
    f.close()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        tokens = line.split()
        if len(tokens) < 3:
            continue
        if tokens[1] != 'nr_file_pages':
            continue

        usage_by_field['time'].append(tokens[0])
        try:
            usage_by_field['nr_file_pages'].append(int(tokens[2]))
        except ValueError:
            usage_by_field['nr_file_pages'].append(None)

    return usage_by_field

def parse_bite_size(fpath):
    f = open(fpath, 'r')
    lines = f.readlines()
    f.close()

    # let insert a placeholder at the start for each operation
    # First get the timestamps'
    start_timestamp = None
    for cur_line in lines:
        cur_line = cur_line.strip()
        if cur_line and cur_line[0].isdigit():
            # if we did not find anything
            break
        if not cur_line.startswith('>>>'):
            continue
        start_timestamp = int(cur_line.split(' ')[-3][:-9])

    all_operations = ['R', 'WS', 'FF', 'RA', 'WFS', 'RM', 'WM', 'W']
    timestamps = []
    op = []
    bite_size = []
    start_sector = []
    num_sectors = []
    for cur_line in lines:
        cur_line = cur_line.strip()
        if not (cur_line and cur_line[0].isdigit()):
            continue
        cur_line = cur_line.split(' ')
        cur_timestamp = int(cur_line[0][:-1][:-9])  # to seconds
        cur_op = cur_line[1][:-1]
        cur_bite_size = int(int(cur_line[2][:-1]) / 1024)  # in KB
        cur_start_sector = int(cur_line[3][:-1])
        cur_num_sectors = int(cur_line[4])
        timestamps.append(cur_timestamp)
        op.append(cur_op)
        bite_size.append(cur_bite_size)
        start_sector.append(cur_start_sector)
        num_sectors.append(cur_num_sectors)

    # now lets add a placeholder for each operation
    if start_timestamp:
        op = all_operations + op
        timestamps = [start_timestamp] * len(all_operations) + timestamps
        bite_size = [4] * len(all_operations) + bite_size
        start_sector = [0] * len(all_operations) + start_sector
        num_sectors = [1] * len(all_operations) + num_sectors

    return timestamps, op, bite_size, start_sector, num_sectors


def aggregate_io_by_sec(timestamp, bite_size):
    timestamp_aggregated = []
    bite_size_aggregated = []
    cur_ts = timestamp[0]
    cur_aggregated_size = 0
    cur_qps = 0

    # for next_ts, next_bite_size in zip(timestamp, bite_size):
    #     if cur_ts == next_ts:
    #         cur_aggregated_size += next_bite_size
    #         cur_qps += 1
    #     else:
    #         timestamp_aggregated.append(cur_ts)
    #         bite_size_aggregated.append(cur_aggregated_size)
    #         # cur_ts += 1
    #         # while cur_ts < next_ts:
    #         #     timestamp_aggregated.append(cur_ts)
    #         #     bite_size_aggregated.append(0)
    #         #     cur_ts += 1
    #         cur_ts = next_ts
    #         cur_aggregated_size = next_bite_size

    results = {}
    for cur_ts, cur_bite_size in zip(timestamp, bite_size):
        if not cur_ts in results:
            results[cur_ts] = 0
        results[cur_ts] += cur_bite_size

    sorted_ts = list(results.keys())
    sorted_ts.sort()

    prev_ts = sorted_ts[0] - 1
    for cur_ts in sorted_ts:
        while prev_ts + 1 < cur_ts:
            timestamp_aggregated.append(prev_ts)
            bite_size_aggregated.append(0)
            prev_ts += 1
        timestamp_aggregated.append(cur_ts)
        bite_size_aggregated.append(results[cur_ts])
        prev_ts = cur_ts

    return timestamp_aggregated, bite_size_aggregated, cur_qps


def aggregate_sector_id():
    pass


def sector_usage_count(start_sector, num_sectors):
    pass


'''
What do we need?
  1. Count of each size
  2. IO traffic by operation
'''


def process_bite_size(timestamps, op, bite_size):
    bite_size_by_op = {
        'ALL_READS': {
            'ts': [],
            'size': []
        },
        'ALL_WRITES': {
            'ts': [],
            'size': []
        }
    }
    bite_size_aggregated = {}
    io_size_by_sec = {}
    for cur_ts, cur_op, cur_bite_size in zip(timestamps, op, bite_size):
        if not (cur_op in bite_size_by_op):
            bite_size_by_op[cur_op] = {'ts': [], 'size': []}
        bite_size_by_op[cur_op]['ts'].append(cur_ts)
        bite_size_by_op[cur_op]['size'].append(cur_bite_size)
        if 'W' in cur_op:
            bite_size_by_op['ALL_WRITES']['ts'].append(cur_ts)
            bite_size_by_op['ALL_WRITES']['size'].append(cur_bite_size)
        if 'R' in cur_op:
            bite_size_by_op['ALL_READS']['ts'].append(cur_ts)
            bite_size_by_op['ALL_READS']['size'].append(cur_bite_size)

    for cur_op in bite_size_by_op.keys():
        bite_size_aggregated[cur_op] = {}
        for cur_size in bite_size_by_op[cur_op]['size']:
            if not (cur_size in bite_size_aggregated[cur_op]):
                bite_size_aggregated[cur_op][cur_size] = 0
            bite_size_aggregated[cur_op][cur_size] += 1

    for cur_op in bite_size_by_op.keys():
        cur_ts_aggregated, cur_size_aggregated, cur_qps = aggregate_io_by_sec(
            bite_size_by_op[cur_op]['ts'], bite_size_by_op[cur_op]['size'])
        io_size_by_sec[cur_op] = {
            'ts': cur_ts_aggregated,
            'size': cur_size_aggregated,
            'qps': cur_qps
        }
        # if cur_op == 'RA':
        #     print(cur_ts_aggregated)
        #     print(len(cur_ts_aggregated))

    return bite_size_by_op, bite_size_aggregated, io_size_by_sec


def get_access_frequency(op, start_sector, num_sectors):
    results = {}
    for cur_op, cur_start_sector, cur_num_sectors in zip(
            op, start_sector, num_sectors):
        if not cur_op in results:
            results[cur_op] = {}
        for cur_sector in range(cur_start_sector,
                                cur_start_sector + cur_num_sectors):
            if not cur_sector in results[cur_op]:
                results[cur_op][cur_sector] = 0
            results[cur_op][cur_sector] += 1

    reverse_index = {}
    for cur_op in results.keys():
        reverse_index[cur_op] = {}
        for cur_sec, cur_num in results[cur_op].items():
            if cur_num not in reverse_index[cur_op]:
                reverse_index[cur_op][cur_num] = 0
            reverse_index[cur_op][cur_num] += 1

    return results, reverse_index

# Example usage:
if __name__ == '__main__':
    metrics = parse_vectordb_bench_output('./test-output.txt')
    print(f"QPS:     {metrics['qps']}")
    print(f"Latency: {metrics['latency']} s (p99)")
    print(f"Recall:  {metrics['recall']}")
