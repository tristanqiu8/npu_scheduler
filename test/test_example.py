# =================== tests/test_examples.py ===================

#!/usr/bin/env python3
"""
演示测试 - 测试所有演示程序正常工作
"""

import pytest
import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import SchedulerConfig


class TestBasicDemo:
    """基础演示测试"""
    
    def test_basic_demo_import(self):
        """测试基础演示导入"""
        try:
            from examples.basic_demo import run_basic_demo, create_basic_tasks
            assert callable(run_basic_demo)
            assert callable(create_basic_tasks)
        except ImportError as e:
            pytest.fail(f"基础演示导入失败: {e}")
    
    def test_basic_demo_execution(self):
        """测试基础演示执行"""
        from examples.basic_demo import run_basic_demo
        
        config = SchedulerConfig.for_testing()
        config.verbose_logging = False  # 安静模式
        
        try:
            scheduler = run_basic_demo(config)
            assert scheduler is not None
            assert len(scheduler.tasks) > 0
        except Exception as e:
            pytest.fail(f"基础演示执行失败: {e}")
    
    def test_basic_demo_tasks_creation(self):
        """测试基础演示任务创建"""
        from examples.basic_demo import create_basic_tasks
        
        tasks = create_basic_tasks()
        assert len(tasks) > 0
        
        # 验证任务有效性
        for task in tasks:
            assert task.task_id is not None
            assert task.name is not None
            assert task.is_valid()


class TestOptimizationDemo:
    """优化演示测试"""
    
    def test_optimization_demo_import(self):
        """测试优化演示导入"""
        try:
            from examples.optimization_demo import run_optimization_demo
            assert callable(run_optimization_demo)
        except ImportError as e:
            pytest.fail(f"优化演示导入失败: {e}")
    
    def test_optimization_demo_without_optimization(self):
        """测试优化演示（无优化模块）"""
        from examples.optimization_demo import run_optimization_demo, OPTIMIZATION_AVAILABLE
        
        if not OPTIMIZATION_AVAILABLE:
            # 应该优雅处理模块缺失
            result = run_optimization_demo()
            assert result is None
        else:
            config = SchedulerConfig.for_testing()
            config.verbose_logging = False
            
            try:
                result = run_optimization_demo(config)
                # 如果有优化模块，应该返回结果
                assert result is not None
            except Exception as e:
                pytest.fail(f"优化演示执行失败: {e}")


class TestTutorialDemo:
    """教程演示测试"""
    
    def test_tutorial_demo_import(self):
        """测试教程演示导入"""
        try:
            from examples.tutorial_demo import run_tutorial_demo, quick_tutorial
            assert callable(run_tutorial_demo)
            assert callable(quick_tutorial)
        except ImportError as e:
            pytest.fail(f"教程演示导入失败: {e}")
    
    def test_quick_tutorial_execution(self):
        """测试快速教程执行"""
        from examples.tutorial_demo import quick_tutorial
        
        try:
            scheduler = quick_tutorial()
            assert scheduler is not None
            assert len(scheduler.tasks) > 0
            assert len(scheduler.schedule_history) > 0
        except Exception as e:
            pytest.fail(f"快速教程执行失败: {e}")
    
    def test_non_interactive_tutorial(self):
        """测试非交互式教程"""
        from examples.tutorial_demo import run_tutorial_demo
        
        try:
            scheduler = run_tutorial_demo(interactive=False)
            assert scheduler is not None
        except Exception as e:
            pytest.fail(f"非交互式教程失败: {e}")


class TestExamplesIntegration:
    """演示集成测试"""
    
    def test_all_examples_basic_functionality(self):
        """测试所有演示基本功能"""
        config = SchedulerConfig.for_testing()
        config.verbose_logging = False
        
        # 测试基础演示
        from examples.basic_demo import run_basic_demo
        basic_scheduler = run_basic_demo(config)
        assert basic_scheduler is not None
        
        # 测试教程演示
        from examples.tutorial_demo import quick_tutorial
        tutorial_scheduler = quick_tutorial()
        assert tutorial_scheduler is not None
        
        # 验证两个调度器都有结果
        assert len(basic_scheduler.schedule_history) > 0
        assert len(tutorial_scheduler.schedule_history) > 0
    
    def test_examples_with_different_configs(self):
        """测试演示在不同配置下的表现"""
        configs = [
            SchedulerConfig.for_production(),
            SchedulerConfig.for_development(),
            SchedulerConfig.for_testing()
        ]
        
        from examples.basic_demo import run_basic_demo
        
        for config in configs:
            config.verbose_logging = False
            try:
                scheduler = run_basic_demo(config)
                assert scheduler is not None
                assert len(scheduler.tasks) > 0
            except Exception as e:
                pytest.fail(f"配置 {config} 下演示失败: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])