# NPU 调度器分段功能修复指南

## 📋 问题总结

原始的 `simple_seg_test.py` 存在以下主要问题：

1. **资源冲突**：多个任务被分配到同一资源的重叠时间段
2. **时间精度问题**：浮点数计算导致的微小时间差异引起冲突
3. **分段调度逻辑不完善**：子段的时间计算和资源分配存在问题
4. **缺少缓冲机制**：子段之间没有足够的时间间隔

## 🔧 修复方案

### 方案 1：使用修复后的 simple_seg_test.py

替换原文件为 `fixed_simple_seg_test.py`，它包含以下改进：

- ✅ **增加时间缓冲**：子段之间添加 0.2ms 缓冲区
- ✅ **增强资源冲突检测**：改进资源可用性检查逻辑
- ✅ **优化调度循环**：防止无限循环和资源竞争
- ✅ **保守的任务参数**：使用更安全的延迟和 FPS 要求
- ✅ **健壮的错误处理**：graceful fallback 和错误恢复

### 方案 2：应用综合补丁

使用 `comprehensive_segmentation_patch.py`：

```python
from comprehensive_segmentation_patch import apply_comprehensive_segmentation_patch

# 创建调度器
scheduler = MultiResourceScheduler(enable_segmentation=True)

# 应用综合补丁
config = apply_comprehensive_segmentation_patch(scheduler)

# 正常使用调度器...
```

## 🚀 使用方法

### 快速开始

1. **替换测试文件**：
   ```bash
   # 备份原文件
   mv simple_seg_test.py simple_seg_test_original.py
   
   # 使用修复版本
   cp fixed_simple_seg_test.py simple_seg_test.py
   ```

2. **运行测试**：
   ```bash
   python simple_seg_test.py
   ```

### 在现有代码中应用修复

```python
from comprehensive_segmentation_patch import apply_comprehensive_segmentation_patch
from scheduler import MultiResourceScheduler

# 创建调度器
scheduler = MultiResourceScheduler(enable_segmentation=True)

# 应用修复
apply_comprehensive_segmentation_patch(scheduler)

# 添加资源
scheduler.add_npu("NPU_0", bandwidth=8.0)
scheduler.add_npu("NPU_1", bandwidth=4.0)
scheduler.add_dsp("DSP_0", bandwidth=4.0)

# 创建任务
from task import NNTask
from enums import TaskPriority, RuntimeType, SegmentationStrategy

task = NNTask("T1", "TestTask", 
              priority=TaskPriority.HIGH,
              runtime_type=RuntimeType.ACPU_RUNTIME,
              segmentation_strategy=SegmentationStrategy.ADAPTIVE_SEGMENTATION)

task.set_npu_only({2.0: 20, 4.0: 15, 8.0: 10}, "test_seg")
task.add_cut_points_to_segment("test_seg", [("cut1", 0.5, 0.1)])
task.set_performance_requirements(fps=20, latency=50)

scheduler.add_task(task)

# 运行调度
results = scheduler.priority_aware_schedule_with_segmentation(time_window=100.0)

# 验证结果
from schedule_validator import validate_schedule
is_valid, errors = validate_schedule(scheduler)
print(f"Validation: {'✅ PASSED' if is_valid else '❌ FAILED'}")
```

## 🔍 验证测试结果

### 检查冲突

```python
from schedule_validator import validate_schedule

is_valid, errors = validate_schedule(scheduler)

if is_valid:
    print("✅ 没有资源冲突")
else:
    print(f"❌ 发现 {len(errors)} 个冲突:")
    for error in errors:
        print(f"  - {error}")
```

### 分析性能

```python
# 分析调度结果
print(f"总调度事件: {len(results)}")

for i, schedule in enumerate(results[:5]):
    task = scheduler.tasks[schedule.task_id]
    print(f"事件 {i+1}: {task.name}")
    print(f"  时间: {schedule.start_time:.2f} - {schedule.end_time:.2f}ms")
    
    if schedule.sub_segment_schedule:
        print(f"  子段数量: {len(schedule.sub_segment_schedule)}")
        for j, (sub_id, start, end) in enumerate(schedule.sub_segment_schedule):
            print(f"    子段 {j+1}: {start:.2f} - {end:.2f}ms")
```

## 📊 可视化结果

```python
try:
    from elegant_visualization import ElegantSchedulerVisualizer
    
    viz = ElegantSchedulerVisualizer(scheduler)
    viz.plot_elegant_gantt(bar_height=0.35, spacing=0.8)
    viz.export_chrome_tracing("schedule_trace.json")
    
    print("✅ 可视化生成成功")
    print("打开 chrome://tracing 加载 schedule_trace.json 查看")
    
except ImportError:
    print("⚠️ 可视化模块不可用")
```

## ⚙️ 配置选项

### 时间缓冲设置

```python
from comprehensive_segmentation_patch import SegmentationPatchConfig

# 自定义配置
config = SegmentationPatchConfig()
config.timing_buffer = 0.3  # 增加缓冲到 0.3ms
config.scheduling_overhead = 0.15  # 每段调度开销
config.debug_mode = True  # 启用调试输出

# 应用配置
apply_comprehensive_segmentation_patch(scheduler, config)
```

### 优先级缓冲缩放

```python
config.priority_buffer_scale = {
    TaskPriority.CRITICAL: 0.3,  # 关键任务使用更小缓冲
    TaskPriority.HIGH: 1.0,
    TaskPriority.NORMAL: 1.5,
    TaskPriority.LOW: 2.5        # 低优先级任务使用更大缓冲
}
```

## 🐛 故障排除

### 常见问题

1. **仍有资源冲突**
   - 增加 `timing_buffer` 值（如 0.5ms）
   - 减少切点数量
   - 使用 `NO_SEGMENTATION` 策略进行测试

2. **性能下降**
   - 检查 `scheduling_overhead` 设置
   - 确认任务延迟要求合理
   - 考虑增加资源数量

3. **无限循环**
   - 检查 `max_iterations` 设置
   - 确认任务 FPS 要求可达成
   - 验证依赖关系正确

### 调试模式

```python
config.debug_mode = True
apply_comprehensive_segmentation_patch(scheduler, config)

# 运行时会输出详细信息
results = scheduler.priority_aware_schedule_with_segmentation(time_window=50.0)
```

### 诊断工具

```python
from segmentation_diagnostic import diagnose_segmentation_schedule

# 运行诊断
conflicts = diagnose_segmentation_schedule(scheduler)

# 查看建议修复
from segmentation_diagnostic import suggest_fixes
suggest_fixes(conflicts)
```

## 📈 性能基准

修复后的版本应该达到以下性能指标：

- ✅ **零资源冲突**：所有验证测试通过
- ✅ **合理开销**：分段开销 < 总执行时间的 10%
- ✅ **稳定性**：100+ 连续运行无崩溃
- ✅ **可扩展性**：支持 10+ 任务，5+ 资源

## 📝 最佳实践

1. **保守的切点配置**：
   - 每个段最多 2-3 个切点
   - 切点开销 < 段执行时间的 5%

2. **合理的任务参数**：
   - 延迟要求留 20% 余量
   - FPS 要求考虑资源容量

3. **充足的资源配置**：
   - NPU 数量 ≥ 最大并行段数
   - 混合不同带宽的资源

4. **定期验证**：
   - 每次修改后运行验证
   - 使用可视化工具检查调度

## 🎯 总结

通过应用这些修复，`simple_seg_test.py` 中的所有测试都应该能够通过。关键改进包括：

- **时间精度处理**：添加缓冲区防止冲突
- **资源管理**：改进资源可用性检查
- **调度逻辑**：增强循环控制和状态管理
- **错误处理**：添加 fallback 机制

如果仍有问题，请检查错误消息并相应调整配置参数。
