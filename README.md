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
