#!/usr/bin/env python3
"""
utils/patches.py - 修复版补丁系统
与当前调度器实现兼容
"""

import functools
from typing import Dict, List, Optional, Any
from core.enums import ResourceType, TaskPriority


class PatchRegistry:
    """补丁注册表"""
    
    def __init__(self):
        self.patches = {}
        self.applied_patches = set()
    
    def register(self, name: str, description: str = ""):
        """注册补丁装饰器"""
        def decorator(patch_func):
            self.patches[name] = {
                'function': patch_func,
                'description': description,
                'applied': False
            }
            return patch_func
        return decorator
    
    def apply_patch(self, scheduler, patch_name: str):
        """应用单个补丁"""
        if patch_name not in self.patches:
            raise ValueError(f"未知补丁: {patch_name}")
        
        if patch_name in self.applied_patches:
            print(f"⚠️  补丁 {patch_name} 已经应用过")
            return
        
        patch_info = self.patches[patch_name]
        try:
            patch_info['function'](scheduler)
            self.applied_patches.add(patch_name)
            patch_info['applied'] = True
            if hasattr(scheduler, 'verbose') and scheduler.verbose:
                print(f"✅ 补丁 {patch_name} 应用成功")
        except Exception as e:
            if hasattr(scheduler, 'verbose') and scheduler.verbose:
                print(f"⚠️  补丁 {patch_name} 跳过: {e}")
            # 不抛出异常，继续执行
    
    def apply_all_patches(self, scheduler):
        """应用所有注册的补丁"""
        for patch_name in self.patches:
            try:
                self.apply_patch(scheduler, patch_name)
            except Exception as e:
                if hasattr(scheduler, 'verbose') and scheduler.verbose:
                    print(f"⚠️  跳过失败的补丁 {patch_name}: {e}")
    
    def list_patches(self):
        """列出所有可用补丁"""
        print("📋 可用补丁列表:")
        for name, info in self.patches.items():
            status = "✅ 已应用" if info['applied'] else "⭕ 未应用"
            print(f"   {name}: {info['description']} [{status}]")


# 全局补丁注册表
patches = PatchRegistry()


@patches.register("basic_scheduler_enhancement", "基础调度器增强")
def patch_basic_scheduler_enhancement(scheduler):
    """基础调度器增强补丁"""
    
    # 添加详细输出控制
    if not hasattr(scheduler, 'verbose'):
        scheduler.verbose = False
    
    def set_verbose(self, verbose: bool):
        """设置详细输出模式"""
        self.verbose = verbose
    
    # 绑定方法到调度器
    import types
    scheduler.set_verbose = types.MethodType(set_verbose, scheduler)
    
    # 添加调试信息收集
    if not hasattr(scheduler, 'debug_info'):
        scheduler.debug_info = {
            'scheduling_decisions': [],
            'performance_metrics': {}
        }
    
    def log_scheduling_decision(self, task_id, decision_type, details):
        """记录调度决策"""
        if self.verbose:
            print(f"🔍 [{decision_type}] 任务 {task_id}: {details}")
        
        self.debug_info['scheduling_decisions'].append({
            'task_id': task_id,
            'type': decision_type,
            'details': details,
            'timestamp': len(self.debug_info['scheduling_decisions'])
        })
    
    scheduler.log_scheduling_decision = types.MethodType(log_scheduling_decision, scheduler)


@patches.register("resource_queue_enhancement", "资源队列增强")
def patch_resource_queue_enhancement(scheduler):
    """资源队列增强补丁"""
    
    # 检查是否有priority_queues属性
    if not hasattr(scheduler, 'priority_queues'):
        if hasattr(scheduler, 'verbose') and scheduler.verbose:
            print("⚠️  调度器没有priority_queues，跳过队列增强")
        return
    
    # 为每个优先级队列添加增强功能
    for queue_id, queue in scheduler.priority_queues.items():
        
        def enhanced_is_available(self, current_time: float) -> bool:
            """增强的可用性检查"""
            # 基础可用性检查
            basic_available = self.available_time <= current_time
            
            # 绑定状态检查
            binding_available = True
            if hasattr(self, 'bound_until'):
                binding_available = self.bound_until <= current_time
            
            return basic_available and binding_available
        
        # 绑定增强方法
        import types
        queue.enhanced_is_available = types.MethodType(enhanced_is_available, queue)


@patches.register("scheduling_algorithm_enhancement", "调度算法增强")
def patch_scheduling_algorithm_enhancement(scheduler):
    """调度算法增强补丁"""
    
    # 保存原始的调度方法
    original_schedule = scheduler.priority_aware_schedule_with_segmentation
    
    @functools.wraps(original_schedule)
    def enhanced_schedule(time_window: float = 1000.0):
        """增强的调度算法"""
        
        if hasattr(scheduler, 'verbose') and scheduler.verbose:
            print(f"🚀 开始增强调度，时间窗口: {time_window}ms")
        
        # 记录开始时间
        import time
        start_time = time.time()
        
        # 调用原始调度算法
        try:
            results = original_schedule(time_window)
            
            # 记录性能
            end_time = time.time()
            execution_time = (end_time - start_time) * 1000  # 转换为毫秒
            
            if hasattr(scheduler, 'debug_info'):
                scheduler.debug_info['performance_metrics']['last_scheduling_time'] = execution_time
                scheduler.debug_info['performance_metrics']['last_scheduled_tasks'] = len(scheduler.schedule_history)
            
            if hasattr(scheduler, 'verbose') and scheduler.verbose:
                print(f"✅ 调度完成，耗时: {execution_time:.2f}ms，调度了 {len(scheduler.schedule_history)} 个任务")
            
            return results
            
        except Exception as e:
            if hasattr(scheduler, 'verbose') and scheduler.verbose:
                print(f"❌ 调度过程中出错: {e}")
            return None
    
    # 替换调度方法
    scheduler.priority_aware_schedule_with_segmentation = enhanced_schedule


@patches.register("performance_monitoring", "性能监控增强")
def patch_performance_monitoring(scheduler):
    """性能监控增强补丁"""
    
    def get_enhanced_performance_metrics(self, time_window: float):
        """获取增强的性能指标"""
        
        # 获取基础指标
        if hasattr(self, 'get_performance_metrics'):
            metrics = self.get_performance_metrics(time_window)
        else:
            # 如果没有基础方法，创建简单的指标
            from core.models import PerformanceMetrics
            metrics = PerformanceMetrics()
            
            if self.schedule_history:
                metrics.total_tasks = len(self.schedule_history)
                metrics.makespan = max(s.end_time for s in self.schedule_history)
                
                latencies = [s.end_time - s.start_time for s in self.schedule_history]
                metrics.average_latency = sum(latencies) / len(latencies)
        
        # 添加调试信息
        if hasattr(self, 'debug_info'):
            debug_metrics = self.debug_info.get('performance_metrics', {})
            if 'last_scheduling_time' in debug_metrics:
                # 可以添加调度器性能相关的指标
                pass
        
        return metrics
    
    # 绑定增强方法
    import types
    scheduler.get_enhanced_performance_metrics = types.MethodType(get_enhanced_performance_metrics, scheduler)


def patch_scheduler(scheduler):
    """应用推荐的补丁组合"""
    recommended_patches = [
        "basic_scheduler_enhancement",
        "resource_queue_enhancement", 
        "scheduling_algorithm_enhancement",
        "performance_monitoring"
    ]
    
    for patch_name in recommended_patches:
        try:
            patches.apply_patch(scheduler, patch_name)
        except Exception as e:
            # 静默处理错误，不影响主流程
            pass


def apply_all_patches(scheduler):
    """应用所有可用补丁"""
    patches.apply_all_patches(scheduler)


def apply_production_patches(scheduler):
    """应用生产环境补丁"""
    production_patches = [
        "basic_scheduler_enhancement",
        "performance_monitoring"
    ]
    
    for patch_name in production_patches:
        if patch_name in patches.patches:
            try:
                patches.apply_patch(scheduler, patch_name)
            except Exception:
                pass  # 静默处理


def apply_development_patches(scheduler):
    """应用开发环境补丁"""
    development_patches = [
        "basic_scheduler_enhancement",
        "resource_queue_enhancement",
        "scheduling_algorithm_enhancement",
        "performance_monitoring"
    ]
    
    for patch_name in development_patches:
        if patch_name in patches.patches:
            try:
                patches.apply_patch(scheduler, patch_name)
            except Exception:
                pass  # 静默处理


def list_available_patches():
    """列出所有可用补丁"""
    patches.list_patches()


def get_patch_status() -> Dict[str, bool]:
    """获取补丁应用状态"""
    return {name: info['applied'] for name, info in patches.patches.items()}


# 添加一个安全的补丁应用函数
def safe_patch_scheduler(scheduler):
    """安全地应用补丁，不会因为错误而中断程序"""
    try:
        patch_scheduler(scheduler)
        if hasattr(scheduler, 'verbose') and scheduler.verbose:
            print("✅ 补丁系统应用完成")
    except Exception as e:
        # 完全静默处理，确保不影响主程序
        pass


if __name__ == "__main__":
    # 测试补丁系统
    print("=== 补丁系统测试 ===")
    list_available_patches()
    
    print(f"\n已注册 {len(patches.patches)} 个补丁")
    print("使用 safe_patch_scheduler(scheduler) 安全应用补丁")