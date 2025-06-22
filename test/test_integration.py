#!/usr/bin/env python3
"""
集成测试 - 测试整个系统协同工作
"""

import pytest
import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core import NNTask, SchedulerFactory
from core.enums import TaskPriority, RuntimeType, ResourceType
from config import SchedulerConfig
from utils import validate_schedule
from visualization import SchedulerVisualizer


class TestEndToEndIntegration:
    """端到端集成测试"""
    
    def test_complete_workflow(self):
        """测试完整工作流程"""
        # 1. 配置创建
        config = SchedulerConfig.for_testing()
        
        # 2. 调度器创建
        scheduler = SchedulerFactory.create_scheduler(config)
        assert len(scheduler.resources[ResourceType.NPU]) > 0
        assert len(scheduler.resources[ResourceType.DSP]) > 0
        
        # 3. 任务创建
        tasks = self._create_integration_task_set()
        for task in tasks:
            scheduler.add_task(task)
        
        assert len(scheduler.tasks) == len(tasks)
        
        # 4. 调度执行
        results = scheduler.priority_aware_schedule_with_segmentation(200.0)
        assert results is not None
        assert len(scheduler.schedule_history) > 0
        
        # 5. 结果验证
        is_valid, errors = validate_schedule(scheduler, verbose=False)
        
        # 检查严重错误
        critical_errors = [e for e in errors if e.error_type == "RESOURCE_CONFLICT"]
        assert len(critical_errors) == 0, f"发现资源冲突: {critical_errors}"
        
        # 6. 可视化测试
        try:
            visualizer = SchedulerVisualizer(scheduler)
            assert visualizer is not None
        except Exception as e:
            pytest.skip(f"可视化测试跳过: {e}")
        
        # 7. 性能指标
        metrics = scheduler.get_performance_metrics(200.0)
        assert metrics.total_tasks > 0
        assert metrics.makespan > 0
    
    def _create_integration_task_set(self):
        """创建集成测试任务集"""
        tasks = []
        
        # 关键任务
        critical_task = NNTask("INT_CRITICAL", "IntegrationCritical",
                              priority=TaskPriority.CRITICAL,
                              runtime_type=RuntimeType.DSP_RUNTIME)
        critical_task.set_npu_only({8.0: 5}, "critical_seg")
        critical_task.set_performance_requirements(fps=100, latency=10)
        tasks.append(critical_task)
        
        # 高优先级任务
        high_task = NNTask("INT_HIGH", "IntegrationHigh",
                          priority=TaskPriority.HIGH,
                          runtime_type=RuntimeType.ACPU_RUNTIME)
        high_task.set_dsp_npu_sequence([
            (ResourceType.DSP, {8.0: 3}, 0, "high_dsp"),
            (ResourceType.NPU, {4.0: 12}, 3, "high_npu")
        ])
        high_task.set_performance_requirements(fps=50, latency=20)
        tasks.append(high_task)
        
        # 普通任务
        normal_task = NNTask("INT_NORMAL", "IntegrationNormal",
                            priority=TaskPriority.NORMAL,
                            runtime_type=RuntimeType.ACPU_RUNTIME)
        normal_task.set_npu_only({2.0: 30}, "normal_seg")
        normal_task.set_performance_requirements(fps=20, latency=50)
        tasks.append(normal_task)
        
        return tasks
    
    def test_configuration_variations(self):
        """测试不同配置变体"""
        configs = [
            SchedulerConfig.for_production(),
            SchedulerConfig.for_development(),
            SchedulerConfig.for_testing()
        ]
        
        for config in configs:
            scheduler = SchedulerFactory.create_scheduler(config)
            assert scheduler is not None
            
            # 添加简单任务
            task = NNTask("CONFIG_TEST", "ConfigTest")
            task.set_npu_only({4.0: 10}, "config_seg")
            scheduler.add_task(task)
            
            # 运行调度
            results = scheduler.priority_aware_schedule_with_segmentation(50.0)
            assert results is not None
    
    def test_error_handling(self):
        """测试错误处理"""
        config = SchedulerConfig.for_testing()
        scheduler = SchedulerFactory.create_scheduler(config)
        
        # 无效任务处理
        invalid_task = NNTask("INVALID", "InvalidTask")
        scheduler.add_task(invalid_task)
        
        # 调度应该处理无效任务而不崩溃
        results = scheduler.priority_aware_schedule_with_segmentation(50.0)
        # 不应该抛出异常
    
    def test_stress_scenario(self):
        """压力测试场景"""
        config = SchedulerConfig.for_testing()
        scheduler = SchedulerFactory.create_scheduler(config)
        
        # 创建多个竞争任务
        for i in range(5):
            task = NNTask(f"STRESS_{i}", f"StressTask{i}",
                         priority=TaskPriority.HIGH)
            task.set_npu_only({4.0: 10}, f"stress_seg_{i}")
            task.set_performance_requirements(fps=50, latency=20)
            scheduler.add_task(task)
        
        # 运行调度
        results = scheduler.priority_aware_schedule_with_segmentation(200.0)
        assert results is not None
        
        # 验证没有严重冲突
        is_valid, errors = validate_schedule(scheduler, verbose=False)
        resource_conflicts = [e for e in errors if e.error_type == "RESOURCE_CONFLICT"]
        
        # 允许一些性能违规，但不允许资源冲突
        assert len(resource_conflicts) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])