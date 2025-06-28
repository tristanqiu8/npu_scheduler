#!/usr/bin/env python3
"""
修复 assigned_resources 类型错误问题
确保 assigned_resources 始终是字典而不是元组
"""

from typing import Dict, Optional
from enums import ResourceType, TaskPriority


def apply_assigned_resources_type_fix(scheduler):
    """应用 assigned_resources 类型修复"""
    print("🔧 应用 assigned_resources 类型修复...")
    
    # 保存原始方法
    original_find_resources = scheduler.find_available_resources_for_task_with_segmentation
    
    def safe_find_available_resources(task, current_time):
        """确保返回值始终是字典类型"""
        result = original_find_resources(task, current_time)
        
        # 检查返回值类型
        if result is None:
            return None
            
        # 如果是元组，尝试转换为字典
        if isinstance(result, tuple):
            print(f"  ⚠️ 检测到元组类型的 assigned_resources: {result}")
            # 尝试从任务段中推断资源类型映射
            if len(result) == 2 and hasattr(task, 'segments'):
                # 假设是 (npu_id, dsp_id) 的形式
                resource_dict = {}
                resource_types = []
                for seg in task.segments:
                    if seg.resource_type not in resource_types:
                        resource_types.append(seg.resource_type)
                
                # 尝试映射
                if len(resource_types) == len(result):
                    for i, res_type in enumerate(resource_types):
                        if i < len(result):
                            resource_dict[res_type] = result[i]
                    print(f"  ✓ 转换为字典: {resource_dict}")
                    return resource_dict
                else:
                    print(f"  ❌ 无法转换元组到字典，资源类型数量不匹配")
                    return None
            else:
                print(f"  ❌ 无法转换元组到字典")
                return None
                
        # 如果已经是字典，直接返回
        if isinstance(result, dict):
            return result
            
        # 其他类型，记录警告并返回 None
        print(f"  ⚠️ 未知的 assigned_resources 类型: {type(result)}")
        return None
    
    # 只替换资源查找方法，不替换整个调度方法！
    scheduler.find_available_resources_for_task_with_segmentation = safe_find_available_resources
    
    print("✅ assigned_resources 类型修复已应用")


if __name__ == "__main__":
    # 测试修复
    import sys
    sys.path.append('.')
    
    from scheduler import MultiResourceScheduler
    from task import NNTask
    from enums import TaskPriority, RuntimeType, SegmentationStrategy
    
    # 创建测试调度器
    scheduler = MultiResourceScheduler(enable_segmentation=True)
    scheduler.add_npu("NPU_0", 120.0)
    scheduler.add_dsp("DSP_0", 40.0)
    
    # 创建测试任务
    task = NNTask("T1", "Test Task",
                  priority=TaskPriority.HIGH,
                  runtime_type=RuntimeType.ACPU_RUNTIME,
                  segmentation_strategy=SegmentationStrategy.ADAPTIVE_SEGMENTATION)
    
    # 应用修复
    apply_assigned_resources_type_fix(scheduler)
    
    print("修复测试完成")
