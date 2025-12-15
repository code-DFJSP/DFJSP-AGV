import random
import json
from pathlib import Path
from typing import List, Tuple

# 参考https://scheduleopt.github.io/benchmarks/fjsplib#:~:text=,options%20%28duration%20machine

OPS_PER_JOB_RANGE = (5, 15)
DURATION_RANGE = (10, 30)

# 输出目录
OUTPUT_DIR = Path("generated_instances")
OUTPUT_DIR.mkdir(exist_ok=True)
RANDOM_SEED = 12345
random.seed(RANDOM_SEED)

def sample_int(rng: Tuple[int, int]) -> int:
    return random.randint(rng[0], rng[1])

def generate_instance(
    jobs: int,
    machines: int,
        agv: int,
    ops_per_job_range: Tuple[int, int],
    options_range: Tuple[int, int],
    duration_range: Tuple[int, int],
) -> Tuple[str, List[List[Tuple[int, int]]]]:
    job_lines = []
    total_ops = 0
    for j in range(jobs):
        ops = sample_int(ops_per_job_range)
        total_ops += ops
        job_ops = []
        for _ in range(ops):
            opt_count = sample_int(options_range)
            machines_list = list(range(1, machines + 1))
            random.shuffle(machines_list)
            chosen_machines = machines_list[:opt_count]
            mach_durations = []
            for m_id in chosen_machines:
                d = sample_int(duration_range)
                mach_durations.append((m_id, d))
            job_ops.append((opt_count, mach_durations))
        job_lines.append(job_ops)
    avg_flex = sum(opt for job in job_lines for opt, _ in job) / total_ops
    header_line = f"{jobs} {machines} {avg_flex:.1f} {agv}"
    return header_line, job_lines

def job_lines_to_text(job_lines: List[List[Tuple[int, List[Tuple[int, int]]]]]) -> List[str]:
    lines = []
    for job_ops in job_lines:
        parts = [str(len(job_ops))]
        for opt_count, mach_durations in job_ops:
            parts.append(str(opt_count))
            for m_id, d in mach_durations:
                parts.append(str(m_id))
                parts.append(str(d))
        lines.append(" ".join(parts))
    return lines

def main():
    instance_idx = 1
    max_instances = 1
    generated = 0
    attempts = 0
    while generated < max_instances:
        attempts += 1
        jobs = 50
        machines = 5
        agv = 2
        OPTIONS_PER_OPERATION_RANGE = (3, machines)
        header, job_lines = generate_instance(
            jobs, machines, agv, OPS_PER_JOB_RANGE,
            OPTIONS_PER_OPERATION_RANGE, DURATION_RANGE
        )
        total_ops = sum(len(job_ops) for job_ops in job_lines)
        if 20 <= total_ops:
            lines = [header] + job_lines_to_text(job_lines)
            fname = OUTPUT_DIR / f"{jobs}_{machines}_{agv}.fjs"
            with open(fname, "w") as f:
                for ln in lines:
                    f.write(ln + "\n")
            print(f"Generated {fname}  jobs={jobs} machines={machines} total_ops={total_ops}")
            instance_idx += 1
            generated += 1
        else:
            continue

    print(f"Done. Attempts={attempts}, Generated={generated}")

if __name__ == "__main__":
    main()
