#!/usr/bin/env python3
"""
核心功能测试 - 合并所有核心模块测试
"""

import pytest
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core import NNTask, MultiResourceScheduler, SchedulerFactory
from core.enums import TaskPriority, RuntimeType, ResourceType, SegmentationStrategy
from core.models import ResourceSegment, ResourceUnit, TaskScheduleInfo
from config import SchedulerConfig
from utils import validate_schedule


class TestEnums:
    """枚举测试"""
    
    def test_task_priority_values(self):
        """测试任务优先级数值"""
        assert TaskPriority.CRITICAL.value == 0
        assert TaskPriority.HIGH.value == 1
        assert TaskPriority.NORMAL.value == 2
        assert TaskPriority.LOW.value == 3
    
    def test_resource_types(self):
        """测试资源类型"""
        assert ResourceType.NPU.value == "NPU"
        assert ResourceType.DSP.value == "DSP"
    
    def test_runtime_types(self):
        """测试运行时类型"""
        assert RuntimeType.DSP_RUNTIME.value == "DSP_Runtime"
        assert RuntimeType.ACPU_RUNTIME.value == "ACPU_Runtime"


class TestModels:
    """数据模型测试"""
    
    def test_resource_segment(self):
        """测试资源段"""
        segment = ResourceSegment(
            resource_type=ResourceType.NPU,
            bandwidth_duration_map={2.0: 40, 4.0: 20, 8.0: 10},
            segment_id="test_segment"
        )
        
        assert segment.get_duration(4.0) == 20
        assert segment.get_min_duration() == 10
        assert segment.get_max_duration() == 40
    
    def test_resource_unit(self):
        """测试资源单元"""
        unit = ResourceUnit("NPU_0", ResourceType.NPU, bandwidth=8.0)
        
        assert unit.is_available(0.0)
        
        unit.reserve("T1", 0.0, 10.0)
        assert unit.available_time == 10.0
        assert not unit.is_available(5.0)
        assert unit.is_available(10.0)
    
    def test_task_schedule_info(self):
        """测试任务调度信息"""
        info = TaskScheduleInfo(
            task_id="T1",
            start_time=0.0,
            end_time=20.0,
            assigned_resources={ResourceType.NPU: "NPU_0"}
        )
        
        assert info.actual_latency == 20.0
        assert info.get_total_execution_time() == 20.0


class TestTask:
    """任务测试"""
    
    def test_task_creation(self):
        """测试任务创建"""
        task = NNTask("T1", "TestTask", priority=TaskPriority.HIGH)
        
        assert task.task_id == "T1"
        assert task.name == "TestTask"
        assert task.priority == TaskPriority.HIGH
    
    def test_npu_only_configuration(self):
        """测试NPU专用配置"""
        task = NNTask("T1", "NPUTask")
        task.set_npu_only({4.0: 20}, "npu_segment")
        
        assert len(task.segments) == 1
        assert task.segments[0].resource_type == ResourceType.NPU
        assert task.segments[0].get_duration(4.0) == 20
    
    def test_dsp_npu_sequence(self):
        """测试DSP+NPU序列"""
        task = NNTask("T1", "SequenceTask")
        task.set_dsp_npu_sequence([
            (ResourceType.DSP, {8.0: 5}, 0, "dsp_seg"),
            (ResourceType.NPU, {4.0: 15}, 5, "npu_seg")
        ])
        
        assert len(task.segments) == 2
        assert task.segments[0].resource_type == ResourceType.DSP
        assert task.segments[1].resource_type == ResourceType.NPU
    
    def test_performance_requirements(self):
        """测试性能需求"""
        task = NNTask("T1", "PerfTask")
        task.set_performance_requirements(fps=30, latency=50)
        
        assert task.fps_requirement == 30
        assert task.latency_requirement == 50
    
    def test_task_validation(self):
        """测试任务验证"""
        task = NNTask("T1", "ValidTask")
        
        # 没有段的任务无效
        assert not task.is_valid()
        
        # 添加段后有效
        task.set_npu_only({4.0: 10}, "valid_segment")
        assert task.is_valid()


class TestScheduler:
    """调度器测试"""
    
    @pytest.fixture
    def scheduler(self):
        """调度器fixture"""
        config = SchedulerConfig.for_testing()
        return SchedulerFactory.create_scheduler(config)
    
    def test_scheduler_creation(self, scheduler):
        """测试调度器创建"""
        assert scheduler is not None
        assert ResourceType.NPU in scheduler.resources
        assert ResourceType.DSP in scheduler.resources
        assert len(scheduler.resources[ResourceType.NPU]) > 0
        assert len(scheduler.resources[ResourceType.DSP]) > 0
    
    def test_task_addition(self, scheduler):
        """测试任务添加"""
        task = NNTask("T1", "TestTask", priority=TaskPriority.HIGH)
        task.set_npu_only({4.0: 10}, "test_segment")
        
        scheduler.add_task(task)
        assert "T1" in scheduler.tasks
    
    def test_basic_scheduling(self, scheduler):
        """测试基础调度"""
        # 创建任务
        task1 = NNTask("T1", "Task1", priority=TaskPriority.HIGH)
        task1.set_npu_only({4.0: 15}, "segment1")
        task1.set_performance_requirements(fps=30, latency=50)
        
        task2 = NNTask("T2", "Task2", priority=TaskPriority.NORMAL)
        task2.set_npu_only({4.0: 20}, "segment2")
        task2.set_performance_requirements(fps=20, latency=60)
        
        scheduler.add_task(task1)
        scheduler.add_task(task2)
        
        # 运行调度
        results = scheduler.priority_aware_schedule_with_segmentation(100.0)
        
        assert results is not None
        assert len(scheduler.schedule_history) > 0
        
        # 验证优先级顺序
        if len(scheduler.schedule_history) >= 2:
            schedules = sorted(scheduler.schedule_history, key=lambda s: s.start_time)
            first_task = scheduler.tasks[schedules[0].task_id]
            assert first_task.priority.value <= TaskPriority.HIGH.value
    
    def test_different_runtime_types(self, scheduler):
        """测试不同运行时类型"""
        # DSP Runtime任务
        dsp_task = NNTask("DSP_T1", "DSPTask", 
                         priority=TaskPriority.HIGH,
                         runtime_type=RuntimeType.DSP_RUNTIME)
        dsp_task.set_npu_only({4.0: 10}, "dsp_seg")
        
        # ACPU Runtime任务
        acpu_task = NNTask("ACPU_T1", "ACPUTask",
                          priority=TaskPriority.NORMAL,
                          runtime_type=RuntimeType.ACPU_RUNTIME)
        acpu_task.set_npu_only({4.0: 15}, "acpu_seg")
        
        scheduler.add_task(dsp_task)
        scheduler.add_task(acpu_task)
        
        results = scheduler.priority_aware_schedule_with_segmentation(100.0)
        
        # 验证两种类型都被调度
        task_ids = [s.task_id for s in scheduler.schedule_history]
        assert "DSP_T1" in task_ids
        assert "ACPU_T1" in task_ids


class TestSchedulerFactory:
    """调度器工厂测试"""
    
    def test_production_config(self):
        """测试生产环境配置"""
        config = SchedulerConfig.for_production()
        scheduler = SchedulerFactory.create_scheduler(config)
        
        assert scheduler is not None
        assert not config.enable_segmentation
        assert config.apply_patches
    
    def test_development_config(self):
        """测试开发环境配置"""
        config = SchedulerConfig.for_development()
        scheduler = SchedulerFactory.create_scheduler(config)
        
        assert scheduler is not None
        assert config.enable_validation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])