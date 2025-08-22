import re
import os
import argparse
import subprocess
import pandas as pd
import concurrent.futures
from tqdm import tqdm
from collections import defaultdict
from openpyxl import load_workbook

######################################## re pattern #################################################

# 支持3种初始化格式
# NCCL INFO ncclCommInitRank comm 0x12026320 rank 0 nranks 2 cudaDev 0 nvmlDev 0 busId 19000 commId 0xc266e719e89eb1ab - Init START
# NCCL INFO ncclCommInitRankConfigMemOpt comm 0x55a7a339db50 rank 0 nranks 8 cudaDev 0 nvmlDev 0 busId 4b000 commId 0x62a8e54c3929a7df - Init START    
# NCCL INFO ncclCommInitRankConfig
initRank_start_pattern = re.compile(
    r'.*?ncclCommInitRank(?:Config)?(?:MemOpt)? comm ([0-9a-fx]+) rank 0 nranks (\d+) cudaDev (\d+) .* commId ([0-9a-fx]+ - Init START)',
    re.IGNORECASE
)

# NCCL INFO comm 0x55a7a339db50 rank 0 nRanks 8 nNodes 1 localRanks 8 localRank 0 MNNVL 0
nNodes_pattern = re.compile(
    r'.*?NCCL INFO comm ([0-9a-fx]+) rank 0 nRanks (\d+) nNodes (\d+) localRanks (\d)+ localRank \d+ MNNVL \d+',
    re.IGNORECASE
)

# NCCL INFO comm 0x555e0a3036b0 rank 0 nranks 8 cudaDev 0 setBuffSizesMemOpt commName paddleComm set LL BuffSize to 524288
commName_pattern = re.compile(
    r'NCCL INFO comm ([0-9a-fx]+) rank 0 nranks \d+ cudaDev \d+ setBuffSizesMemOpt commName (\w+) set LL BuffSize to (\d+)',
    re.IGNORECASE
)

# NCCL INFO comm 0x55a7a339db50, 16 coll channels, 16 collnet channels, 0 nvls channels, 16 p2p channels, 2 p2p channels per peer
nChannels_pattern = re.compile(
    r'.*?NCCL INFO comm ([0-9a-fx]+), (\d+) coll channels, (\d+) collnet channels, (\d+) nvls channels, (\d+) p2p channels, (\d+) p2p channels per peer',
    re.IGNORECASE
)

# NCCL INFO Init timings - ncclCommInitRankMemOpt: rank 0 nranks 8 total 0.48 (kernels 0.13, alloc 0.19, bootstrap 0.00, allgathers 0.01, topo 0.11, graphs 0.00, connections 0.04, rest 0.00)


# INFO(NCCL_TUNING, "%s: %ld Bytes -> Algo %s proto %s channel{Lo..Hi}={%d..%d} comm %p"
tuning_pattern = re.compile(
   r'.*?NCCL INFO (AllReduce|Broadcast|Reduce|AllGather|ReduceScatter|Send|Recv): (\d+) Bytes -> Algo (\w+) proto (\w+) channel\{Lo\.\.Hi\}=\{(\d+)\.\.(\d+)\} comm ([0-9a-fx]+)'
)

def new_comm_group():
    return {
        'id': 0,
        'name': '',
        'commId': '',
        'rank0_comm': '',
        'nranks': '',
        'nNodes': 0,
        'cudaDev': '',
        'localRanks': 0,
        'nchannels': 0,
        'initTime': 0,
        # colls[coll][msgsize] = [ {algo, proto, nc_used, count}, ... ]
        'colls': defaultdict(lambda: defaultdict(list))
    }

def parse_nccl_log(log_content):
    status = defaultdict(new_comm_group)
    commid_to_rank0comm = {}  # commId -> rank0_comm

    lines = log_content.strip().split("\n")
    for line in tqdm(lines, desc="log parsing", ncols=80):
        line = line.strip()
    
        # 1.匹配通信组初始化行
        init_match = initRank_start_pattern.search(line)
        if init_match:
            rank0_comm = init_match.group(1)
            nranks = int(init_match.group(2))
            cudaDev = init_match.group(3)
            comm_id = init_match.group(4)
            commid_to_rank0comm[comm_id] = rank0_comm

            group = status[comm_id]
            group['id'] = len(status)
            group['name'] = '-'
            group['commId'] = comm_id
            group['rank0_comm'] = rank0_comm
            group['nranks'] = nranks
            group['cudaDev'] = cudaDev #此处仅为rank 0的cudaDev,后续需要补全，这个通信组的所有rank的cudaDev
            continue

        # 2.匹配 nNodes 行
        nNodes_match = nNodes_pattern.search(line)
        if nNodes_match:
            rank0_comm = nNodes_match.group(1)
            nranks = int(nNodes_match.group(2))
            nNodes = int(nNodes_match.group(3))
            localRanks = int(nNodes_match.group(4))
            # 查找 comm_id， 找不到comm_id，跳过
            comm_id = next((cid for cid, r0c in commid_to_rank0comm.items() if r0c == rank0_comm), None)
            if comm_id is None:
                continue

            group = status[comm_id]
            group['nNodes'] = nNodes
            group['localRanks'] = localRanks
            # print(f"[nNodes_match] id={group['id']},rank0_comm={rank0_comm}, nNodes={nNodes}, localRanks={localRanks}, comm_id={comm_id}")
            continue

        # 3.匹配 commName 行
        commName_match = commName_pattern.search(line)
        if commName_match:
            rank0_comm = commName_match.group(1)
            commNameStr = commName_match.group(2)
            comm_id = next((cid for cid, r0c in commid_to_rank0comm.items() if r0c == rank0_comm), None)
            if comm_id is None:
                continue

            group = status[comm_id]
            group['name'] = commNameStr
            continue

        # 4.匹配 nChannels 行
        nc_match= nChannels_pattern.search(line)
        if(nc_match):
            rank0_comm = nc_match.group(1)
            nc = int(nc_match.group(2))
            comm_id = next((cid for cid, r0c in commid_to_rank0comm.items() if r0c == rank0_comm), None)
            if comm_id is None:
                continue

            group = status[comm_id]
            group['nchannels'] = nc
            # print(f"[nChannels_match] id={group['id']},rank0_comm={rank0_comm}, nc={nc}, comm_id={comm_id}")
            continue

        # 5.匹配tuning行，获取算法/协议/通道等信息
        tuning_match = tuning_pattern.search(line)
        if tuning_match:
            coll=tuning_match.group(1)
            msgsize = int(tuning_match.group(2))
            algo = tuning_match.group(3)
            proto = tuning_match.group(4)
            channel_lo = int(tuning_match.group(5))
            channel_hi = int(tuning_match.group(6))
            rank0_comm = tuning_match.group(7)
            nc_used = channel_hi - channel_lo + 1
            if algo == 'Unknown':
                algo = '-'

            comm_id = next((cid for cid, r0c in commid_to_rank0comm.items() if r0c == rank0_comm), None)
            if comm_id is None:
                continue
            group = status[comm_id]
            
            # 查找是否已有该msgsize+ algo+proto+nc_used的统计项
            entry_list = group['colls'][coll][msgsize]
            entry = next((e for e in entry_list if e['algo'] == algo and e['proto'] == proto and e['nc_used'] == nc_used), None)
            if entry:
                entry['count'] += 1
            else:
                entry_list.append({'algo': algo, 'proto': proto, 'nc_used': nc_used, 'count': 1})

    return status

def print_status(status, base_name=None, aggregate=False):
    col_widths = {
        "no": 4, "name":14, "commId": 20, "rank0 comm": 20, "nranks": 8, "nNodes": 8, "Dev": 4,"localRanks": 10,
        "nchannels": 10, "coll": 14, "msgsize/B": 14, "min_size": 12, "max_size": 12,
        "algo": 6, "proto": 8, "nc_used": 8, "count": 6
    }
    def pad(val, width):
        return str(val).ljust(width)[:width]
    formatters = {col: (lambda w: (lambda x: pad(x, w)))(width) for col, width in col_widths.items()}
    
    # 生成表格数据
    rows = []
    for comm_id, group in status.items():
        first_group = True
        for coll, msgsize_dict in group['colls'].items():
            if aggregate:
                # 聚合: (algo, proto, nc_used) -> {'count':..., 'min':..., 'max':...}
                agg = {}
                for msgsize, entry_list in msgsize_dict.items():
                    for entry in entry_list:
                        key = (entry['algo'], entry['proto'], entry['nc_used'])
                        if key not in agg:
                            agg[key] = {
                                'count': 0,
                                'min_size': msgsize,
                                'max_size': msgsize
                            }
                        agg[key]['count'] += entry['count']
                        agg[key]['min_size'] = min(agg[key]['min_size'], msgsize)
                        agg[key]['max_size'] = max(agg[key]['max_size'], msgsize)
                for (algo, proto, nc_used), info in agg.items():
                    rows.append({
                        "no": group['id'] if first_group else "",
                        "name": group['name'] if first_group else "",
                        "commId": comm_id if first_group else "",
                        "rank0 comm": group['rank0_comm'] if first_group else "",
                        "nranks": group['nranks'] if first_group else "",
                        "nNodes": group['nNodes'] if first_group else "",
                        "Dev": group['cudaDev'] if first_group else "",
                        "localRanks": group['localRanks'] if first_group else "",
                        "nchannels": group['nchannels'] if first_group else "",
                        "coll": coll,
                        "min_size": info['min_size'],
                        "max_size": info['max_size'],
                        "algo": algo,
                        "proto": proto,
                        "nc_used": nc_used,
                        "count": info['count']
                    })
                    first_group = False
            else:
                for msgsize in sorted(msgsize_dict.keys()):# 对每个coll 下的 msgsize 进行升序排序
                    entry_list = msgsize_dict[msgsize]
                    for entry in entry_list:
                        rows.append({
                            "no": group['id'] if first_group else "",# 只输出1次通信组基础信息
                            "name": group['name'] if first_group else "",
                            "commId": comm_id if first_group else "",
                            "rank0 comm": group['rank0_comm'] if first_group else "",
                            "nranks": group['nranks'] if first_group else "",
                            "nNodes": group['nNodes'] if first_group else "",
                            "Dev": group['cudaDev'] if first_group else "",
                            "localRanks": group['localRanks'] if first_group else "",
                            "nchannels": group['nchannels'] if first_group else "",
                            "coll": coll,
                            "msgsize/B": msgsize,
                            "algo": entry['algo'],
                            "proto": entry['proto'],
                            "nc_used": entry['nc_used'],
                            "count": entry['count']
                        })
                        first_group = False
    df = pd.DataFrame(rows)

    header = " ".join([pad(col, col_widths[col]) for col in df.columns])
    lines = []
    lines.append(f"\nNCCL日志统计报告—logfile {base_name}\n")
    lines.append("=" * len(header))
    lines.append(header)
    lines.append("-" * len(header))
    # print(df.to_string(index=False, header=False, justify='left', formatters=formatters))

    last_comm_id = None
    for idx, row in df.iterrows():
        # 仅在通信组切换时打印分隔线（跳过第一行）
        if last_comm_id is not None and row['commId'] != '' and row['commId'] != last_comm_id:
            lines.append("- " * (len(header)//2))
        # 打印当前行
        lines.append(" ".join([pad(str(row[col]), col_widths[col]) for col in df.columns]))
        if row['commId'] != '':
            last_comm_id = row['commId']

    lines.append("=" * len(header))
        
    # if base_name:
    #     excel_file = base_name + ".xlsx"
    #     df.to_excel(excel_file, index=False)
    #     wb = load_workbook(excel_file)
    #     ws = wb.active
    #     for idx, col in enumerate(df.columns, 1):
    #         width = col_widths.get(col, 12)
    #         ws.column_dimensions[chr(64+idx)].width = width
    #     wb.save(excel_file)
    #     print(f"\n已导出到Excel: {excel_file}\n")
    
    return lines

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
    parser.add_argument("-d", "--dir", help="log目录,处理目录下所有.log文件")
    parser.add_argument("-o", "--output", nargs='?', const='', help="可选,若不指定则为result.txt")
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
