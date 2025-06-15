# npu_scheduler

A General Framework for Multiple AI tasks and Multiple NPU core Schedulers

# 文件结构说明：

enums.py - 枚举定义

ResourceType: 资源类型 (NPU/DSP)
TaskPriority: 任务优先级 (CRITICAL/HIGH/NORMAL/LOW)

models.py - 数据模型和基础结构

ResourceSegment: 资源使用段
ResourceUnit: 资源单元
TaskScheduleInfo: 任务调度信息

task.py - 神经网络任务类

NNTask: 完整的任务定义和管理

priority_queue.py - 优先级队列管理

ResourcePriorityQueues: 单个资源的优先级队列

scheduler.py - 多资源调度器

MultiResourceScheduler: 核心调度算法和逻辑

visualization.py - 可视化功能

SchedulerVisualizer: 所有图表生成功能

main.py - 主程序和使用示例

完整的使用示例和演示代码

# 主要改进：

* ✅ 模块化设计: 每个文件职责单一，便于维护和测试
* ✅ 清晰的依赖关系: 从底层模型到上层应用的层次结构
* ✅ 易于扩展: 可以独立修改和扩展每个模块
* ✅ 重用性好: 各个类可以在不同项目中独立使用
* ✅ 文档完善: 包含详细的README说明文档

现在你可以：

独立运行 python main.py 来查看完整示例
单独导入需要的模块，如 from scheduler import MultiResourceScheduler
轻松地修改或扩展特定功能而不影响其他部分
更好地进行单元测试和代码维护

每个文件都保持了原有的功能完整性，同时提供了更好的代码组织结构。

# Runtime升级 2025/06/15

## 最后更新主程序模块## 模块化升级完成！

现在我已经基于你原有的模块化代码结构，逐个模块进行了升级，添加了运行时配置功能：

### 升级的模块：

1. **enums.py** - 新增 `RuntimeType` 枚举
2. **models.py** - 新增 `ResourceBinding` 数据模型，升级 `TaskScheduleInfo`
3. **task.py** - 为 `NNTask` 添加运行时配置支持
4. **priority_queue.py** - 为优先级队列添加资源绑定功能
5. **scheduler.py** - 核心调度器添加运行时感知调度算法
6. **visualization.py** - 新的可视化模块支持运行时配置显示
7. **main.py** - 增强的主程序演示运行时配置

### 新增功能特性：

#### 🔄 **两种运行时模式**

* **DSP_Runtime** : 资源耦合绑定，执行期间不可被打断
* **ACPU_Runtime** : 默认流水线模式，允许资源共享

#### 🎯 **智能调度策略**

```python
# 根据运行时类型自动选择调度策略
if task.runtime_type == RuntimeType.DSP_RUNTIME and task.requires_resource_binding():
    return self.find_bound_resources(task, current_time)  # 绑定调度
else:
    return self.find_pipelined_resources(task, current_time)  # 流水线调度
```

#### 📊 **增强的可视化**

* 不同图案表示运行时类型（斜线 = 绑定，实心 = 流水线）
* 蓝色虚线显示资源绑定时间段
* 任务标签包含运行时标识（B=绑定，P=流水线）

#### 🔍 **运行时性能分析**

* 对比不同运行时的调度频率和延迟
* 资源绑定效率统计
* 运行时类型分布分析

这种模块化升级方式保持了原有代码结构的清晰性，每个模块职责明确，便于维护和进一步扩展。你可以根据需要单独使用或修改任何一个模块。
