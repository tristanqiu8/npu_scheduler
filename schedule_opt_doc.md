## 总结

我现在创建了一个完整的调度优化框架，将所有调度相关的因素都作为可优化的变量：

### 核心概念

1. **SchedulingDecisionVariable**: 包含所有决策变量
   - `priority`: 任务优先级（CRITICAL/HIGH/NORMAL/LOW）
   - `runtime_type`: 运行时类型（DSP_Runtime/ACPU_Runtime）
   - `segmentation_configs`: 每个段的切分配置索引
   - `core_assignments`: 每个段分配到的具体核心ID

2. **SchedulingSearchSpace**: 定义每个任务的搜索空间
   - 允许的优先级列表
   - 允许的运行时类型列表
   - 每个段的可选切分配置
   - 可用的核心资源

3. **SchedulingOptimizer**: 优化器
   - 支持贪心算法和遗传算法
   - 可配置的目标函数（延迟、吞吐量、利用率等）
   - 支持约束条件

### 主要特性

1. **联合优化**: 同时优化所有变量，而不是独立优化
2. **灵活的搜索空间**: 可以为每个任务定义不同的约束
3. **多目标优化**: 平衡延迟、吞吐量、资源利用率等多个目标
4. **约束支持**: 可以添加特定约束（如某些任务必须是高优先级）

### 使用示例

```python
# 创建优化器
optimizer = SchedulingOptimizer(scheduler)

# 定义搜索空间
optimizer.define_search_space("T1", SchedulingSearchSpace(
    task_id="T1",
    allowed_priorities=[TaskPriority.CRITICAL, TaskPriority.HIGH],
    allowed_runtime_types=[RuntimeType.ACPU_RUNTIME],
    segmentation_options={"vision_seg": [0, 1, 2, 3, 4]},  # 5种配置
    available_cores={ResourceType.NPU: ["NPU_0", "NPU_1", "NPU_2", "NPU_3"]}
))

# 运行优化
solution = optimizer.optimize_greedy(time_window=500.0, iterations=10)
```

这个框架真正将调度问题建模为一个组合优化问题，其中所有的调度决策都是变量，可以通过不同的优化算法来找到最优解。