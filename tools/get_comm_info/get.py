import os
import re
import argparse
import subprocess
import concurrent.futures
from utils import parse_nccl_log, print_status

def process_log_file(log_path, agg):
    # 检查是否是rank 0
    result = subprocess.run(
        ["grep", "rank 0 nrank", log_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        encoding="utf-8"
    )
    if result.returncode != 0:
        print(f"\n[跳过] rank 0 not found - {log_path}\n")
        return
    
    cudaDev = None
    for line in result.stdout.splitlines():
        match = re.search(r'cudaDev (\d+)', line)
        if match:
            cudaDev = match.group(1)
            break

    if cudaDev is not None:
        print(f"\nrank 0 hit, cudaDev: {cudaDev} - {log_path}\n")
    else:
        print
        print(f"\nrank 0's cudaDev was not found - {log_path}\n")
        return 

    base_name = os.path.splitext(os.path.basename(log_path))[0]
    # 拼接cudaDev到文件名
    if cudaDev is not None and base_name:
        base_name = f"{base_name}-cudaDev-{cudaDev}"
    
    with open(log_path, 'r') as f:
        status = parse_nccl_log(f.read())
        lines=print_status(status, base_name, aggregate=agg)
        return base_name, lines

def main():
    parser = argparse.ArgumentParser(description="NCCL通信组特征分析工具")
    parser.add_argument("-i", "--input", help="logfile path")
    parser.add_argument("-d", "--dir", help="log目录，处理目录下所有.log文件")
    parser.add_argument("-o", "--output", nargs='?', const='', help="可选，若不指定则为result.txt")
    parser.add_argument("-a", "--agg", action="store_true", help="是否聚合msgsize区间（algo/proto/nc_used相同则合并）")
    args = parser.parse_args()

    output_file = args.output
    if output_file == '':
        output_file = "result.txt"
    elif output_file is None:
        output_file = None

    if args.dir: 
        log_tasks = []
        for fname in os.listdir(args.dir):
            if fname.endswith('.log'):
                log_path = os.path.join(args.dir, fname)
                log_tasks.append((log_path, args.agg))
        # 多进程并行处理
        results = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_log_file, log_path, agg) for log_path, agg in log_tasks]
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        results.append(result)  # (base_name, lines)
                except Exception as exc:
                    print(f"日志处理出错: {exc}")

        # 按 base_name 排序
        results.sort(key=lambda x: x[0])

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                for base_name, lines in results:
                    for line in lines:
                        # print(line)
                        f.write(line + "\n")
            print(f"\n所有日志已汇总到 {output_file}\n")
    elif args.input:
        # 单文件处理
        base_name,lines= process_log_file(args.input, args.agg)
        for line in lines:
            print(line)

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                for line in lines:
                    f.write(line + "\n")
            print(f"\n已导出到: {output_file}\n")
    else:
        print("请指定 -i <logfile> 或 -d <logdir>")
        exit(1)


if __name__ == "__main__":
    main()
