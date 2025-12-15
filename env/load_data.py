import os
import json

def parse_fjs_line_format(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        raise ValueError("Empty dataset")
    header = lines[0].split()
    num_jobs = int(header[0])
    num_machines = int(header[1])
    num_agvs = int(header[3])
    jobs = []
    for i in range(1, 1 + num_jobs):
        toks = lines[i].split()
        p = 0
        num_ops = int(toks[p]); p += 1
        ops = []
        total_proc = 0.0
        for _ in range(num_ops):
            c = int(toks[p]); p += 1
            candidates = []
            for _ in range(c):
                m = int(toks[p]); p += 1
                t = float(toks[p]); p += 1
                candidates.append({'machine': int(m)-1, 'proc_time': float(t)})
                total_proc += float(t)
            ops.append({'candidates': candidates})
        slack = 0.3
        due_date = total_proc * (1.0 + slack) if total_proc>0 else 100.0
        jobs.append({'ops': ops, 'due_date': due_date})
    return {'num_jobs': num_jobs, 'num_machines': num_machines, 'jobs': jobs, 'num_agvs': num_agvs}

def load_dataset(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        return parse_fjs_line_format(path)
