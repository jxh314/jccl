import re

# 支持两种初始化格式
# NCCL INFO ncclCommInitRank comm 0x12026320 rank 0 nranks 2 cudaDev 0 nvmlDev 0 busId 19000 commId 0xc266e719e89eb1ab - Init START
# NCCL INFO ncclCommInitRankMemOpt comm 0x55a7a339db50 rank 0 nranks 8 cudaDev 0 nvmlDev 0 busId 4b000 commId 0x62a8e54c3929a7df - Init START
initRank_start_pattern = re.compile(
    r'.*?ncclCommInitRank(?:MemOpt)?(?:Config)? comm ([0-9a-fx]+) rank 0 nranks (\d+) cudaDev (\d+) .* commId ([0-9a-fx]+ - Init START)',
    re.IGNORECASE
)

# NCCL INFO comm 0x55a7a339db50 rank 0 nRanks 8 nNodes 1 localRanks 8 localRank 0 MNNVL 0
nNodes_pattern = re.compile(
    r'.*?NCCL INFO comm ([0-9a-fx]+) rank 0 nRanks (\d+) nNodes (\d+) localRanks (\d)+ localRank \d+ MNNVL \d+',
    re.IGNORECASE
)

# NCCL INFO comm 0x555e0a3036b0 rank 0 nranks 8 cudaDev 0 getCommBuffMemAlloc comm->commType 0 commTypeStr NCCL_TP_COMM_MEM_ALLOC
commType_pattern = re.compile(
    r'.*?NCCL INFO comm ([0-9a-fx]+) rank 0 nRanks \d+ cudaDev \d+ getCommBuffMemAlloc comm->commType \d+ commTypeStr NCCL_(.*?)_COMM_MEM_ALLOC',
    re.IGNORECASE
)

# NCCL INFO comm 0x55a7a339db50, 16 coll channels, 16 collnet channels, 0 nvls channels, 16 p2p channels, 2 p2p channels per peer
nChannels_pattern = re.compile(
    r'.*?NCCL INFO comm ([0-9a-fx]+), (\d+) coll channels, (\d+) collnet channels, (\d+) nvls channels, (\d+) p2p channels, (\d+) p2p channels per peer',
    re.IGNORECASE
)

# NCCL INFO Init timings - ncclCommInitRankMemOpt: rank 0 nranks 8 total 0.48 (kernels 0.13, alloc 0.19, bootstrap 0.00, allgathers 0.01, topo 0.11, graphs 0.00, connections 0.04, rest 0.00)


# NCCL INFO Broadcast: opCount 0 sendbuff 0x7f5b6dec9900 recvbuff 0x7f5b6dec9900 count 8 datatype 0 op 0 root 0 comm 0x1e7ae110 [nranks=2] stream 0x1df54ef0
# NCCL INFO Send: opCount 782f sendbuff (nil) recvbuff 0x7f59a1a8fa00 count 2097152 datatype 9 op 0 root 3 comm 0x18076b90 [nranks=4] stream 0xb617630
sendrecv_addr_pattern = r'(?:[0-9a-fx]+|\(nil\))'
coll_pattern = re.compile(
    rf'.*?NCCL INFO (AllReduce|Broadcast|Reduce|AllGather|ReduceScatter|Send|Recv): opCount ([0-9a-fA-Fx]+) '
    rf'sendbuff {sendrecv_addr_pattern} recvbuff {sendrecv_addr_pattern} count (\d+) '
    r'datatype (\d)+ op \d+ root \d+ comm ([0-9a-fx]+) \[nranks=(\d+)\]'
)

# NCCL INFO AllReduce: 4 Bytes -> Algo TREE proto LL channel{Lo..Hi}={0..0} comm 0x55a7a339db50
# INFO(NCCL_TUNING, "%s: %ld Bytes -> Algo %s proto %s channel{Lo..Hi}={%d..%d} comm %p"
tuning_pattern = re.compile(
   r'.*?NCCL INFO (AllReduce|Broadcast|Reduce|AllGather|ReduceScatter|Send|Recv): (\d+) Bytes -> Algo (\w+) proto (\w+) channel\{Lo\.\.Hi\}=\{(\d+)\.\.(\d+)\} comm ([0-9a-fx]+)'
)
