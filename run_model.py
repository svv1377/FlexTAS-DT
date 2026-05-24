"""
使用训练好的 DRL 模型对单个网络实例进行调度并分析结果。
用法: python run_model.py
"""
import logging
import os
import time

from definitions import OUT_DIR, LOG_DIR, DATA_DIR
from src.network.net import generate_graph, FlowGenerator, Network, Link
from src.app.drl_scheduler import DrlScheduler
from src.app.scheduler import ResAnalyzer
from src.lib.log_config import log_config


def dump_topology(network: Network, filename: str):
    """将拓扑信息完整输出到日志文件。"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("网络拓扑信息 (Topology Info)\n")
        f.write("=" * 60 + "\n\n")

        graph = network.graph

        # 输出图基本信息
        f.write(f"节点总数: {len(graph.nodes)}\n")
        f.write(f"有向边总数: {len(graph.edges)}\n")
        f.write(f"线图节点数 (链路数): {len(network.line_graph.nodes)}\n")
        f.write(f"线图边数: {len(network.line_graph.edges)}\n")
        f.write("\n")

        # 输出所有节点及其属性
        f.write("-" * 60 + "\n")
        f.write("节点列表 (Nodes)\n")
        f.write("-" * 60 + "\n")
        for node in graph.nodes:
            attrs = dict(graph.nodes[node])
            f.write(f"  Node: {node}\n")
            for key, value in attrs.items():
                f.write(f"    {key}: {value}\n")
        f.write("\n")

        # 输出所有边及其属性
        f.write("-" * 60 + "\n")
        f.write("有向边列表 (Edges)\n")
        f.write("-" * 60 + "\n")
        for u, v in graph.edges:
            attrs = dict(graph.edges[u, v])
            f.write(f"  Edge: ({u}) -> ({v})\n")
            for key, value in attrs.items():
                f.write(f"    {key}: {value}\n")
        f.write("\n")

        # 输出所有 Link 对象的详细信息（含 gcl_capacity 和 transmission_time）
        f.write("-" * 60 + "\n")
        f.write("链路详情 (Link Details)\n")
        f.write("-" * 60 + "\n")
        for link_id, link in network.links_dict.items():
            f.write(f"  Link ID: {link.link_id}\n")
            f.write(f"    link_rate      : {link.link_rate} Mbps\n")
            f.write(f"    gcl_capacity   : {link.gcl_capacity}\n")
            f.write(f"    trans_time(MTU): {link.transmission_time(1522)} μs\n")
        f.write("\n")

        # 输出线图的邻接关系
        f.write("-" * 60 + "\n")
        f.write("线图邻接关系 (Line Graph Adjacency): 链路 → 相邻链路\n")
        f.write("-" * 60 + "\n")
        for node in network.line_graph.nodes:
            neighbors = list(network.line_graph.neighbors(node))
            f.write(f"  Link {node} <-> {neighbors}\n")

    logging.info(f"拓扑信息已保存到 {filename}")


def dump_flows(network: Network, filename: str):
    """将流信息完整输出到日志文件。"""
    with open(filename, 'w', encoding='utf-8') as f:
        flows = network.flows

        f.write("=" * 60 + "\n")
        f.write("流信息 (Flow Info)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"流总数: {len(flows)}\n\n")

        for i, flow in enumerate(flows):
            f.write(f"[{i}] {flow}\n")
            f.write(f"    flow_id   : {flow.flow_id}\n")
            f.write(f"    src_id    : {flow.src_id}\n")
            f.write(f"    dst_id    : {flow.dst_id}\n")
            f.write(f"    period    : {flow.period} μs\n")
            f.write(f"    payload   : {flow.payload} B\n")
            f.write(f"    e2e_delay : {flow.e2e_delay} μs\n")
            f.write(f"    jitter    : {flow.jitter} μs\n")
            f.write(f"    path      : {flow.path}\n")
            f.write(f"    num_hops  : {len(flow.path)}\n")
            f.write("\n")

    logging.info(f"流信息已保存到 {filename}")


def main():
    # 配置日志（第一个参数是日志文件名，第二个是级别）
    log_config(f"{LOG_DIR}/run_model.log", logging.INFO)

    # ========================
    # 1. 生成网络拓扑和流量
    # ========================
    TOPO = "CEV"           # 拓扑: RRG / ERG / BAG
    NUM_FLOWS = 50         # 流数量
    SEED = 42              # 随机种子
    LINK_RATE = 100        # 链路速率 (Mbps)
    JITTERS = [0.1]        # 抖动约束 (0.1 = ±10%)
    TIMEOUT = 300          # 调度超时 (秒)

    print(f"生成拓扑: {TOPO}, 流数量: {NUM_FLOWS}")
    graph = generate_graph(TOPO, LINK_RATE)
    flow_generator = FlowGenerator(graph, seed=SEED, jitters=JITTERS)
    flows = flow_generator(NUM_FLOWS)
    network = Network(graph, flows)

    print(f"  节点数: {len(network.graph.nodes)}")
    print(f"  链路数: {len(network.links_dict)}")
    print(f"  流数量: {len(network.flows)}")

    # 输出拓扑和流信息到 out/data/
    #timestamp = time.strftime("%Y%m%d_%H%M%S")
    topo_file = os.path.join(DATA_DIR, f"topology_{TOPO}_{SEED}.log")
    flow_file = os.path.join(DATA_DIR, f"flows_{TOPO}_{NUM_FLOWS}_{SEED}.log")
    dump_topology(network, topo_file)
    dump_flows(network, flow_file)
    print(f"拓扑信息已保存: {topo_file}")
    print(f"流信息已保存:   {flow_file}")

    # ========================
    # 2. 创建调度器并加载模型
    # ========================
    print("\n加载 DRL 模型...")
    scheduler = DrlScheduler(network, timeout_s=TIMEOUT)
    scheduler.load_model("model/best_model.zip", "MaskablePPO")

    # ========================
    # 3. 执行调度
    # ========================
    print("开始调度...")
    if scheduler.schedule():
        print("✅ 调度成功！")

        # 获取调度结果
        res = scheduler.get_res()

        # ========================
        # 4. 分析结果
        # ========================
        analyzer = ResAnalyzer(network, res)
        stats = analyzer.analyze()

        print("\n========== 调度结果分析 ==========")
        print(f"链路利用率 - 最小: {stats['link_utilization_min']:.4f}")
        print(f"链路利用率 - 最大: {stats['link_utilization_max']:.4f}")
        print(f"链路利用率 - 平均: {stats['link_utilization_avg']:.4f}")
        print(f"链路利用率 - 标准差: {stats['link_utilization_std']:.4f}")

        print(f"\nGCL 长度 - 最小: {stats['gcl_min']}")
        print(f"GCL 长度 - 最大: {stats['gcl_max']}")
        print(f"GCL 长度 - 平均: {stats['gcl_avg']:.1f}")
        print(f"GCL 长度 - 标准差: {stats['gcl_std']:.1f}")

        print(f"\n端到端延迟 - 最小: {stats['e2e_delay_min']} μs")
        print(f"端到端延迟 - 最大: {stats['e2e_delay_max']} μs")
        print(f"端到端延迟 - 平均: {stats['e2e_delay_avg']:.1f} μs")
        print(f"端到端延迟 - 标准差: {stats['e2e_delay_std']:.1f} μs")

        print(f"\n抖动 - 最小: {stats['jitter_min']} μs")
        print(f"抖动 - 最大: {stats['jitter_max']} μs")
        print(f"抖动 - 平均: {stats['jitter_avg']:.1f} μs")
        print(f"抖动比率 - 平均: {stats['jitter_ratio_avg']:.4f}")

        print(f"\n调度结果日志已保存到 out/ 目录")
    else:
        print("❌ 调度失败，在超时时间内未找到可行解。")


if __name__ == '__main__':
    main()