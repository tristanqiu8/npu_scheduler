#!/usr/bin/env python3
"""
tests/__init__.py - 测试模块入口
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

__version__ = "2.0.0"

# =================== tests/test_scheduler.py ===================

#!/usr/bin/env python3
"""
测试调度器核心功能
"""

import unittest
import sys
import os

# 添加项目根目录
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core import MultiResourceScheduler, NNTask, SchedulerFactory
from core.enums import TaskPriority, RuntimeType, ResourceType, SegmentationStrategy
from config import SchedulerConfig
from utils import validate_schedule


class TestScheduler(unittest.TestCase):
    """调度器测试类"""
    
    def setUp(self):
        """测试前设置"""
        self.config = SchedulerConfig.for_testing()
        self.scheduler = SchedulerFactory.create_scheduler(self.config)
    
    def test_scheduler_creation(self):
        """测试调度器创建"""
        self.assertIsNotNone(self.scheduler)
        self.assertIn(ResourceType.NPU, self.scheduler.resources)
        self.assertIn(ResourceType.DSP, self.scheduler.resources)
        self.assertGreater(len(self.scheduler.resources[ResourceType.NPU]), 0)
        self.assertGreater(len(self.scheduler.resources[ResourceType.DSP]), 0)
    
    def test_task_addition(self):
        """测试任务添加"""
        task = NNTask("T1", "TestTask", 
                     priority=TaskPriority.HIGH,
                     runtime_type=RuntimeType.ACPU_RUNTIME)
        task.set_npu_only({4.0: 10}, "test_segment")
        
        self.scheduler.add_task(task)
        self.assertIn("T1", self.scheduler.tasks)
        self.assertEqual(self.scheduler.tasks["T1"].name, "TestTask")
    
    def test_basic_scheduling(self):
        """测试基础调度功能"""
        # 创建简单任务
        task1 = NNTask("T1", "Task1", priority=TaskPriority.HIGH)
        task1.set_npu_only({4.0: 15}, "segment1")
        task1.set_performance_requirements(fps=30, latency=50)
        
        task2 = NNTask("T2", "Task2", priority=TaskPriority.NORMAL)
        task2.set_npu_only({4.0: 20}, "segment2")
        task2.set_performance_requirements(fps=20, latency=60)
        
        # 添加任务
        self.scheduler.add_task(task1)
        self.scheduler.add_task(task2)
        
        # 运行调度
        results = self.scheduler.priority_aware_schedule_with_segmentation(200.0)
        
        # 验证结果
        self.assertIsNotNone(results)
        self.assertGreater(len(self.scheduler.schedule_history), 0)
        
        # 验证优先级顺序（高优先级任务应该先执行）
        schedules = sorted(self.scheduler.schedule_history, key=lambda s: s.start_time)
        if len(schedules) >= 2:
            first_task = self.scheduler.tasks[schedules[0].task_id]
            self.assertEqual(first_task.priority, TaskPriority.HIGH)
    
    def test_schedule_validation(self):
        """测试调度验证"""
        # 创建任务
        task = NNTask("T1", "ValidationTest", priority=TaskPriority.CRITICAL)
        task.set_npu_only({4.0: 10}, "val_segment")
        task.set_performance_requirements(fps=50, latency=25)
        
        self.scheduler.add_task(task)
        
        # 运行调度
        results = self.scheduler.priority_aware_schedule_with_segmentation(100.0)
        self.assertIsNotNone(results)
        
        # 验证调度结果
        is_valid, errors = validate_schedule(self.scheduler, verbose=False)
        
        # 基本验证应该通过（至少没有资源冲突）
        resource_conflicts = [e for e in errors if e.error_type == "RESOURCE_CONFLICT"]
        self.assertEqual(len(resource_conflicts), 0, "不应该有资源冲突")
    
    def test_different_runtime_types(self):
        """测试不同运行时类型"""
        # DSP Runtime 任务
        dsp_task = NNTask("DSP_T1", "DSPTask", 
                         priority=TaskPriority.HIGH,
                         runtime_type=RuntimeType.DSP_RUNTIME)
        dsp_task.set_dsp_npu_sequence([
            (ResourceType.DSP, {8.0: 5}, 0, "dsp_seg"),
            (ResourceType.NPU, {4.0: 10}, 5, "npu_seg")
        ])
        
        # ACPU Runtime 任务
        acpu_task = NNTask("ACPU_T1", "ACPUTask",
                          priority=TaskPriority.NORMAL,
                          runtime_type=RuntimeType.ACPU_RUNTIME)
        acpu_task.set_npu_only({4.0: 15}, "acpu_seg")
        
        # 添加任务
        self.scheduler.add_task(dsp_task)
        self.scheduler.add_task(acpu_task)
        
        # 运行调度
        results = self.scheduler.priority_aware_schedule_with_segmentation(200.0)
        
        # 验证两种类型的任务都被调度
        task_ids = [s.task_id for s in self.scheduler.schedule_history]
        self.assertIn("DSP_T1", task_ids)
        self.assertIn("ACPU_T1", task_ids)
    
    def test_resource_utilization(self):
        """测试资源利用率计算"""
        # 创建多个任务以提高资源利用率
        for i in range(3):
            task = NNTask(f"T{i}", f"Task{i}", priority=TaskPriority.NORMAL)
            task.set_npu_only({4.0: 10}, f"segment{i}")
            self.scheduler.add_task(task)
        
        # 运行调度
        results = self.scheduler.priority_aware_schedule_with_segmentation(100.0)
        self.assertIsNotNone(results)
        
        # 检查是否有调度历史
        self.assertGreater(len(self.scheduler.schedule_history), 0)
        
        # 计算资源利用率
        if hasattr(self.scheduler, 'get_resource_utilization'):
            total_time = max(s.end_time for s in self.scheduler.schedule_history)
            utilization = self.scheduler.get_resource_utilization(total_time)
            self.assertIsInstance(utilization, dict)


class TestSchedulerFactory(unittest.TestCase):
    """调度器工厂测试类"""
    
    def test_production_config(self):
        """测试生产环境配置"""
        config = SchedulerConfig.for_production()
        scheduler = SchedulerFactory.create_scheduler(config)
        
        self.assertIsNotNone(scheduler)
        self.assertFalse(config.enable_segmentation)  # 生产环境禁用分段
        self.assertTrue(config.apply_patches)         # 应用补丁
    
    def test_development_config(self):
        """测试开发环境配置"""
        config = SchedulerConfig.for_development()
        scheduler = SchedulerFactory.create_scheduler(config)
        
        self.assertIsNotNone(scheduler)
        self.assertTrue(config.enable_validation)     # 启用验证
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = SchedulerConfig(
            enable_segmentation=True,
            enable_validation=True,
            apply_patches=False,
            verbose_logging=True
        )
        scheduler = SchedulerFactory.create_scheduler(config)
        
        self.assertIsNotNone(scheduler)


if __name__ == "__main__":
    # 运行测试
    unittest.main(verbosity=2)


# =================== tests/test_task.py ===================

#!/usr/bin/env python3
"""
测试任务类功能
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core import NNTask
from core.enums import TaskPriority, RuntimeType, ResourceType, SegmentationStrategy


class TestNNTask(unittest.TestCase):
    """神经网络任务测试类"""
    
    def test_task_creation(self):
        """测试任务创建"""
        task = NNTask("T1", "TestTask", 
                     priority=TaskPriority.HIGH,
                     runtime_type=RuntimeType.ACPU_RUNTIME)
        
        self.assertEqual(task.task_id, "T1")
        self.assertEqual(task.name, "TestTask")
        self.assertEqual(task.priority, TaskPriority.HIGH)
        self.assertEqual(task.runtime_type, RuntimeType.ACPU_RUNTIME)
    
    def test_npu_only_configuration(self):
        """测试NPU专用配置"""
        task = NNTask("T1", "NPUTask")
        bandwidth_map = {2.0: 40, 4.0: 20, 8.0: 10}
        
        task.set_npu_only(bandwidth_map, "npu_segment")
        
        self.assertEqual(len(task.segments), 1)
        segment = task.segments[0]
        self.assertEqual(segment.resource_type, ResourceType.NPU)
        self.assertEqual(segment.bandwidth_duration_map, bandwidth_map)
        self.assertEqual(segment.segment_id, "npu_segment")
    
    def test_dsp_npu_sequence(self):
        """测试DSP+NPU序列配置"""
        task = NNTask("T1", "SequenceTask")
        
        sequence = [
            (ResourceType.DSP, {8.0: 5}, 0, "dsp_segment"),
            (ResourceType.NPU, {4.0: 15}, 5, "npu_segment")
        ]
        
        task.set_dsp_npu_sequence(sequence)
        
        self.assertEqual(len(task.segments), 2)
        
        # 检查DSP段
        dsp_segment = task.segments[0]
        self.assertEqual(dsp_segment.resource_type, ResourceType.DSP)
        self.assertEqual(dsp_segment.start_time, 0)
        
        # 检查NPU段
        npu_segment = task.segments[1]
        self.assertEqual(npu_segment.resource_type, ResourceType.NPU)
        self.assertEqual(npu_segment.start_time, 5)
    
    def test_performance_requirements(self):
        """测试性能需求设置"""
        task = NNTask("T1", "PerfTask")
        
        task.set_performance_requirements(fps=30, latency=50)
        
        self.assertEqual(task.fps_requirement, 30)
        self.assertEqual(task.latency_requirement, 50)
    
    def test_segmentation_configuration(self):
        """测试分段配置"""
        task = NNTask("T1", "SegmentedTask",
                     segmentation_strategy=SegmentationStrategy.CUSTOM_SEGMENTATION)
        
        task.set_npu_only({4.0: 20}, "main_segment")
        
        