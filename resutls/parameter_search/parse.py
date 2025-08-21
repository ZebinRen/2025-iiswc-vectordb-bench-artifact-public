import re

def parse_vectordb_bench_output(log_path):
    """
    Parses a log file for the line containing Metric(...) and returns
    a dict with qps, latency (serial_latency_p99), and recall as floats.
    Latency is in ms.

    Parameters
    ----------
    log_path : str
        Path to the log file.

    Returns
    -------
    dict
        {'qps': float, 'latency': float, 'recall': float}

    Raises
    ------
    ValueError
        If no matching Metric line is found.
    """
    # regex to grab qps, serial_latency_p99 and recall
    pattern = re.compile(
        r"qps=(?P<qps>[\d\.]+).*?"
        r"serial_latency_p99=np\.float64\((?P<latency>[\d\.]+)\).*?"
        r"recall=np\.float64\((?P<recall>[\d\.]+)\)")

    with open(log_path, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                return {
                    'qps': float(m.group('qps')),
                    'latency': float(m.group('latency')) * 1000,
                    'recall': float(m.group('recall')),
                }

    raise ValueError(f"No performance metrics found in {log_path!r}")
