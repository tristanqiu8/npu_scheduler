# 迁移指南 - NPU Scheduler 2.0

本指南帮助从旧版本迁移到新的重构版本 (v2.0)。

## 🔄 主要变更概览

### 文件结构变更
```
旧版本 → 新版本
├── main.py → main.py (统一入口)
├── main_fixed.py → examples/fixed_demo.py
├── optimizer_demo.py → examples/optimization_demo.py
├── elegant_visualization.py → visualization/elegant_viz.py
├── visualization.py → [废弃，使用elegant_viz]
├── scheduler_patch.py → utils/patches.py
├── schedule_validator.py → utils/validator.py
└── [新增] config.py, core/scheduler_factory.py
```

### 导入路径变更

#### 核心模块
```python
# 旧版本
from scheduler import MultiResourceScheduler
from task import NNTask
from enums import TaskPriority, ResourceType

# 新版本
from core import MultiResourceScheduler, NNTask
from core.enums import TaskPriority, ResourceType
```

#### 可视化模块
```python
# 旧版本
from elegant_visualization import ElegantSchedulerVisualizer

# 新版本
from visualization import SchedulerVisualizer  # 默认使用elegant风格
```

#### 工具模块
```python
# 旧版本
from scheduler_patch import patch_scheduler
from schedule_validator import validate_schedule

# 新版本
from utils.patches import patch_scheduler
from utils.validator import validate_schedule
```

## 📋 具体迁移步骤

### 1. 更新导入语句

**旧代码:**
```python
from scheduler import MultiResourceScheduler
from task import NNTask
from enums import TaskPriority, ResourceType, RuntimeType
from elegant_visualization import ElegantSchedulerVisualizer
from scheduler_patch import patch_scheduler
```

**新代码:**
```python
from core import MultiResourceScheduler, NNTask, SchedulerFactory
from core.enums import TaskPriority, ResourceType, RuntimeType
from visualization import SchedulerVisualizer
from utils.patches import patch_scheduler
from config import SchedulerConfig
```

### 2. 调度器创建方式

**旧代码:**
```python
# 创建调度器
scheduler = MultiResourceScheduler(enable_segmentation=False)

# 应用补丁
patch_scheduler(scheduler)
```

**新代码:**
```python
# 使用配置和工厂模式
config = SchedulerConfig.for_production()
scheduler = SchedulerFactory.create_scheduler(config)
# 补丁已自动应用
```

### 3. 可视化调用

**旧代码:**
```python
visualizer = ElegantSchedulerVisualizer(scheduler)
visualizer.plot_elegant_gantt(bar_height=0.35, spacing=0.8)
visualizer.export_chrome_tracing("trace.json")
```

**新代码:**
```python
visualizer = SchedulerVisualizer(scheduler)  # 默认elegant风格
visualizer.plot_elegant_gantt(bar_height=0.35, spacing=0.8)
visualizer.export_chrome_tracing("trace.json")
```

### 4. 验证功能

**旧代码:**
```python
from schedule_validator import validate_schedule

is_valid, errors = validate_schedule(scheduler)
```

**新代码:**
```python
from utils.validator import validate_schedule

is_valid, errors = validate_schedule(scheduler, verbose=True)
```

### 5. 主程序结构

**旧代码 (main_fixed.py):**
```python
def main():
    scheduler = MultiResourceScheduler(enable_segmentation=False)
    patch_scheduler(scheduler)
    
    # 创建任务...
    # 运行调度...
    # 可视化...

if __name__ == "__main__":
    main()
```

**新代码:**
```python
from examples import FixedDemo
from config import SchedulerConfig

def main():
    config = SchedulerConfig.for_production()
    demo = FixedDemo(config)
    demo.run()

if __name__ == "__main__":
    main()
```

## 🔧 配置系统迁移

### 硬编码配置 → 配置类

**旧代码:**
```python
scheduler = MultiResourceScheduler(enable_segmentation=False)
patch_scheduler(scheduler)
# 各种硬编码设置...
```

**新代码:**
```python
# 使用预设配置
config = SchedulerConfig.for_production()  # 稳定性优先
# 或
config = SchedulerConfig.for_development()  # 功能完整
# 或
config = SchedulerConfig.for_testing()  # 快速测试

scheduler = SchedulerFactory.create_scheduler(config)
```

### 自定义配置

**新版本提供灵活的配置选项:**
```python
config = SchedulerConfig(
    enable_segmentation=False,      # 生产环境禁用分段
    enable_validation=True,         # 启用验证
    apply_patches=True,             # 自动应用补丁
    visualization_style='elegant',  # 可视化风格
    export_chrome_tracing=True,     # 导出追踪
    verbose_logging=False,          # 生产环境简化日志
    default_time_window=500.0,      # 默认时间窗口
    max_optimization_iterations=5   # 优化迭代次数
)
```

## 🚀 新功能使用

### 命令行界面

**新版本提供统一的命令行界面:**
```bash
# 替代旧的 python main_fixed.py
python main.py --mode fixed --config production

# 替代旧的 python optimizer_demo.py  
python main.py --mode optimization --verbose

# 新的快速测试模式
python main.py --mode basic --config testing --quiet
```

### 演示类系统

**可以继承BaseDemo创建自定义演示:**
```python
from examples import BaseDemo

class MyCustomDemo(BaseDemo):
    def get_demo_name(self):
        return "My Custom Demo"
    
    def get_demo_description(self):
        return "自定义演示功能"
    
    def create_tasks(self):
        # 创建自定义任务...
        return tasks
```

## ⚠️ 注意事项

### 废弃功能
- ❌ `visualization.py` (旧的复杂可视化) → 使用 `visualization/elegant_viz.py`
- ❌ 直接创建调度器 → 使用 `SchedulerFactory`
- ❌ 手动应用补丁 → 配置中自动处理

### 兼容性保持
- ✅ 所有核心调度算法保持不变
- ✅ 任务定义和配置方式兼容
- ✅ 性能特性和优化功能保持
- ✅ Chrome Tracing 功能增强

### 性能改进
- 🚀 更快的调度器初始化
- 🚀 优化的可视化渲染
- 🚀 改进的内存使用
- 🚀 更好的错误处理

## 🔍 迁移检查清单

### 代码更新
- [ ] 更新所有导入语句
- [ ] 使用新的调度器创建方式
- [ ] 更新可视化调用
- [ ] 更新验证函数调用
- [ ] 使用配置系统替代硬编码

### 文件组织
- [ ] 移动自定义演示到 `examples/` 目录
- [ ] 更新测试文件路径
- [ ] 检查资源文件路径

### 功能验证
- [ ] 运行基础功能测试
- [ ] 验证可视化正常工作
- [ ] 确认性能无回归
- [ ] 测试新的命令行界面

### 部署更新
- [ ] 更新启动脚本
- [ ] 更新配置文件
- [ ] 更新文档和说明

## 💡 迁移提示

1. **逐步迁移**: 建议先迁移一个模块，确认无问题后再继续
2. **保留备份**: 迁移前请备份原始代码
3. **测试驱动**: 每个迁移步骤后运行测试确认功能正常
4. **使用新特性**: 充分利用新的配置系统和工厂模式的便利性

## 🆘 遇到问题？

如果在迁移过程中遇到问题：

1. 查看新版本的示例代码 (`examples/` 目录)
2. 参考单元测试了解正确用法
3. 检查配置选项是否正确设置
4. 确认所有依赖包版本正确

## 📞 获取帮助

- 📖 查看完整文档: `README.md`
- 🐛 报告问题: GitHub Issues
- 💬 讨论: GitHub Discussions
- 📧 邮件支持: npu-scheduler@example.com

---

**🎉 迁移完成后，你将获得更好的代码组织、更强的功能和更高的可维护性！**
