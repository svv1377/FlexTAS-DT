~~~
app/                        ← 入口脚本层（训练/测试/绘图）
  train.py                  ← DRL 训练入口
  test.py                   ← 单次测试入口
  plot_training.py          ← 训练曲线可视化

src/
  agent/                    ← Agent 层：特征提取 & GNN 编码
    encoder.py              ← FeaturesExtractor (GIN + MLP)
  env/                      ← 环境层：Gymnasium 强化学习环境
    env.py                  ← NetEnv / TrainingNetEnv / _StateEncoder
  app/                      ← 应用层：调度器 & 评估
    scheduler.py            ← BaseScheduler / ResAnalyzer（基类 & 结果分析）
    drl_scheduler.py        ← DrlScheduler（DRL 调度器封装）
    no_wait_tabu_scheduler.py ← TimeTablingScheduler（No-Wait/Tabu 基线）
    Oliver2018_scheduler.py ← Oliver2018 SMT 基线（Z3 约束求解）
    smt_scheduler.py        ← SmtScheduler（通用 SMT 调度器）
    evaluation.py           ← 评估框架（多调度器对比实验）
  network/                  ← 网络层：拓扑生成 & 网络元素定义
    net.py                  ← Flow / Link / Net / Network / FlowGenerator / 拓扑生成
  lib/                      ← 工具层
    config.py               ← ConfigManager（INI 配置读取，单例）
    execute.py              ← execute_from_command_line（CLI 参数自动解析）
    graph.py                ← neighbors_within_distance（图邻居计算）
    operation.py            ← Operation 数据类 & 流隔离约束检查
    timing_decorator.py     ← 计时装饰器
    log_config.py           ← 日志配置

model/                      ← 存放训练好的模型文件
tests/                      ← 单元测试（与 src 结构镜像）
definitions.py              ← 全局路径常量（ROOT_DIR, OUT_DIR, LOG_DIR 等）
~~~

~~~
                          ┌─────────────────────┐
 flow_feature (拼 link_feature) ──→ MLP(Linear+ReLU) ──→ array_embedding (64)
                          └─────────────────────┘

                          ┌─────────────────────┐
 adjacency_matrix + features_matrix ──→ GINConv×2 + BN + GlobalMeanPool ──→ graph_embedding (64)
                          └─────────────────────┘

                          ┌─────────────────────┐
 remain_hops ──→ MLP(Linear+ReLU) ──→ array_embedding (64)
                          └─────────────────────┘

                               ↓ concat ↓
                        features_dim = 64+64+64 = 192
~~~

~~~
        trans=34, prop=1, proc=[1,6], sync=0

CMRIU2 ──── trans ──── SW22 ──── trans ──── SW7 ──── trans ──── SW31 ──── trans ──── LCM1
   │                    │                    │                    │                    │
   ▼                    ▼                    ▼                    ▼                    ▼
[841,965]             [877,1006]          [1042,1047]          [1083,1088]         [1083,1212]
  无门控               门控@1006            门控@1047            无门控              到达!
 (offset=841         (阻断抖动             (阻断抖动           (wait=124
  因为前序流)          累积)                 累积)               end=1246)
  ~~~