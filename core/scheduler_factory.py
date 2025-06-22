#!/usr/bin/env python3
"""
调度器工厂模块 - 修复版
提供统一的调度器创建和配置功能，避免补丁兼容性问题
"""

from typing import Optional, List, Dict, Any
from .scheduler import MultiResourceScheduler
from .enums import ResourceType


class SchedulerFactory:
    """调度器工厂类"""
    
    @staticmethod
    def create_scheduler(config) -> MultiResourceScheduler:
        """根据配置创建调度器实例"""
        # 创建基础调度器
        scheduler = MultiResourceScheduler(
            enable_segmentation=config.enable_segmentation
        )
        
        # 应用补丁修复（安全模式）
        if config.apply_patches:
            SchedulerFactory._apply_patches_safely(scheduler)
        
        # 设置详细输出
        if hasattr(scheduler, 'set_verbose'):
            scheduler.set_verbose(config.verbose_logging)
        elif hasattr(config, 'verbose_logging'):
            # 直接设置verbose属性
            scheduler.verbose = config.verbose_logging
        
        return scheduler
    
    @staticmethod
    def _apply_patches_safely(scheduler: MultiResourceScheduler):
        """安全地应用调度器补丁修复"""
        try:
            from utils.patches import safe_patch_scheduler
            safe_patch_scheduler(scheduler)
        except ImportError:
            # 如果补丁模块不可用，创建基本的增强
            SchedulerFactory._apply_basic_enhancements(scheduler)
        except Exception as e:
            # 静默处理任何补丁错误
            if hasattr(scheduler, 'verbose') and scheduler.verbose:
                print(f"⚠️  补丁应用时出现问题: {e}")
    
    @staticmethod
    def _apply_basic_enhancements(scheduler: MultiResourceScheduler):
        """应用基本增强功能（不依赖补丁系统）"""
        
        # 添加verbose属性
        if not hasattr(scheduler, 'verbose'):
            scheduler.verbose = False
        
        # 添加set_verbose方法
        def set_verbose(verbose: bool):
            scheduler.verbose = verbose
        
        scheduler.set_verbose = set_verbose
        
        # 添加调试信息收集
        scheduler.debug_info = {
            'scheduling_decisions': [],
            'performance_metrics': {}
        }
        
        # 添加日志记录方法
        def log_scheduling_decision(task_id, decision_type, details):
            if scheduler.verbose:
                print(f"🔍 [{decision_type}] 任务 {task_id}: {details}")
            
            scheduler.debug_info['scheduling_decisions'].append({
                'task_id': task_id,
                'type': decision_type,
                'details': details
            })
        
        scheduler.log_scheduling_decision = log_scheduling_decision
    
    @staticmethod
    def create_test_scheduler(
        num_npu: int = 4,
        num_dsp: int = 2,
        enable_segmentation: bool = False
    ) -> MultiResourceScheduler:
        """创建用于测试的调度器"""
        
        scheduler = MultiResourceScheduler(enable_segmentation=enable_segmentation)
        
        # 清除默认资源
        scheduler.resources = {ResourceType.NPU: [], ResourceType.DSP: []}
        
        # 添加测试资源
        from .models import ResourceUnit
        
        # 添加NPU资源
        for i in range(num_npu):
            npu = ResourceUnit(f"NPU_{i}", ResourceType.NPU, bandwidth=4.0)
            scheduler.add_resource(npu)
        
        # 添加DSP资源  
        for i in range(num_dsp):
            dsp = ResourceUnit(f"DSP_{i}", ResourceType.DSP, bandwidth=8.0)
            scheduler.add_resource(dsp)
        
        # 应用基本增强
        SchedulerFactory._apply_basic_enhancements(scheduler)
        
        return scheduler
    
    @staticmethod
    def create_high_performance_scheduler() -> MultiResourceScheduler:
        """创建高性能配置的调度器"""
        
        scheduler = MultiResourceScheduler(enable_segmentation=False)
        
        # 清除默认资源
        scheduler.resources = {ResourceType.NPU: [], ResourceType.DSP: []}
        
        from .models import ResourceUnit
        
        # 高性能NPU配置
        for i in range(4):
            bandwidth = 8.0 if i < 2 else 4.0  # 前两个高带宽
            npu = ResourceUnit(f"NPU_{i}", ResourceType.NPU, bandwidth=bandwidth)
            scheduler.add_resource(npu)
        
        # 高性能DSP配置
        for i in range(2):
            dsp = ResourceUnit(f"DSP_{i}", ResourceType.DSP, bandwidth=16.0)
            scheduler.add_resource(dsp)
        
        # 应用性能优化
        SchedulerFactory._apply_basic_enhancements(scheduler)
        
        return scheduler
    
    @staticmethod
    def create_minimal_scheduler() -> MultiResourceScheduler:
        """创建最小配置的调度器（用于快速测试）"""
        
        scheduler = MultiResourceScheduler(enable_segmentation=False)
        
        # 清除默认资源
        scheduler.resources = {ResourceType.NPU: [], ResourceType.DSP: []}
        
        from .models import ResourceUnit
        
        # 最小资源配置
        npu = ResourceUnit("NPU_0", ResourceType.NPU, bandwidth=4.0)
        dsp = ResourceUnit("DSP_0", ResourceType.DSP, bandwidth=8.0)
        
        scheduler.add_resource(npu)
        scheduler.add_resource(dsp)
        
        # 应用基本增强
        SchedulerFactory._apply_basic_enhancements(scheduler)
        
        return scheduler
    
    @staticmethod
    def create_scheduler_from_spec(resource_spec: Dict[str, Any]) -> MultiResourceScheduler:
        """根据资源规格创建调度器"""
        
        scheduler = MultiResourceScheduler(
            enable_segmentation=resource_spec.get('enable_segmentation', False)
        )
        
        # 清除默认资源
        scheduler.resources = {ResourceType.NPU: [], ResourceType.DSP: []}
        
        from .models import ResourceUnit
        
        # 根据规格添加NPU资源
        if 'npu_units' in resource_spec:
            for unit_spec in resource_spec['npu_units']:
                npu = ResourceUnit(
                    unit_spec['id'],
                    ResourceType.NPU,
                    bandwidth=unit_spec.get('bandwidth', 4.0)
                )
                scheduler.add_resource(npu)
        
        # 根据规格添加DSP资源
        if 'dsp_units' in resource_spec:
            for unit_spec in resource_spec['dsp_units']:
                dsp = ResourceUnit(
                    unit_spec['id'],
                    ResourceType.DSP,
                    bandwidth=unit_spec.get('bandwidth', 8.0)
                )
                scheduler.add_resource(dsp)
        
        # 应用补丁（如果指定）
        if resource_spec.get('apply_patches', True):
            SchedulerFactory._apply_basic_enhancements(scheduler)
        
        return scheduler


# 预定义的调度器配置
PREDEFINED_CONFIGS = {
    'default': {
        'npu_units': [
            {'id': 'NPU_0', 'bandwidth': 2.0},
            {'id': 'NPU_1', 'bandwidth': 4.0},
            {'id': 'NPU_2', 'bandwidth': 4.0},
            {'id': 'NPU_3', 'bandwidth': 8.0}
        ],
        'dsp_units': [
            {'id': 'DSP_0', 'bandwidth': 4.0},
            {'id': 'DSP_1', 'bandwidth': 8.0}
        ],
        'enable_segmentation': False,
        'apply_patches': True
    },
    
    'high_performance': {
        'npu_units': [
            {'id': 'NPU_0', 'bandwidth': 8.0},
            {'id': 'NPU_1', 'bandwidth': 8.0},
            {'id': 'NPU_2', 'bandwidth': 4.0},
            {'id': 'NPU_3', 'bandwidth': 4.0}
        ],
        'dsp_units': [
            {'id': 'DSP_0', 'bandwidth': 16.0},
            {'id': 'DSP_1', 'bandwidth': 16.0}
        ],
        'enable_segmentation': False,
        'apply_patches': True
    },
    
    'minimal': {
        'npu_units': [
            {'id': 'NPU_0', 'bandwidth': 4.0}
        ],
        'dsp_units': [
            {'id': 'DSP_0', 'bandwidth': 8.0}
        ],
        'enable_segmentation': False,
        'apply_patches': True
    }
}


def create_predefined_scheduler(config_name: str) -> MultiResourceScheduler:
    """创建预定义配置的调度器"""
    if config_name not in PREDEFINED_CONFIGS:
        raise ValueError(f"未知的预定义配置: {config_name}。可用配置: {list(PREDEFINED_CONFIGS.keys())}")
    
    return SchedulerFactory.create_scheduler_from_spec(PREDEFINED_CONFIGS[config_name])


if __name__ == "__main__":
    # 测试工厂功能
    print("=== 调度器工厂测试 ===")
    
    # 测试预定义配置
    for config_name in PREDEFINED_CONFIGS.keys():
        print(f"\n创建 {config_name} 配置调度器...")
        try:
            scheduler = create_predefined_scheduler(config_name)
            
            npu_count = len(scheduler.resources[ResourceType.NPU])
            dsp_count = len(scheduler.resources[ResourceType.DSP])
            
            print(f"  NPU数量: {npu_count}")
            print(f"  DSP数量: {dsp_count}")
            print(f"  分段功能: {'启用' if scheduler.enable_segmentation else '禁用'}")
            print(f"  详细模式: {'启用' if hasattr(scheduler, 'verbose') and scheduler.verbose else '禁用'}")
            
        except Exception as e:
            print(f"  ❌ 创建失败: {e}")
    
    print("\n✅ 工厂测试完成")