#!/usr/bin/env python3
"""
修复可视化中的利用率计算
覆盖scheduler的get_resource_utilization方法
"""

from collections import defaultdict


def fix_scheduler_utilization_calculation(scheduler):
    """修复调度器的资源利用率计算"""
    
    print("🔧 修复资源利用率计算...")
    
    # 保存原始方法
    original_get_utilization = scheduler.get_resource_utilization
    
    def correct_get_resource_utilization(time_window: float):
        """正确计算资源利用率"""
        
        resource_busy_time = defaultdict(float)
        
        # 遍历所有调度历史
        for schedule in scheduler.schedule_history:
            if hasattr(schedule, 'sub_segment_schedule') and schedule.sub_segment_schedule:
                # 处理分段任务
                for sub_seg_id, start_time, end_time in schedule.sub_segment_schedule:
                    # 找到对应的资源
                    task = scheduler.tasks[schedule.task_id]
                    for ss in task.get_sub_segments_for_scheduling():
                        if ss.sub_id == sub_seg_id:
                            if ss.resource_type in schedule.assigned_resources:
                                resource_id = schedule.assigned_resources[ss.resource_type]
                                duration = end_time - start_time
                                resource_busy_time[resource_id] += duration
                            break
            else:
                # 处理非分段任务
                task = scheduler.tasks[schedule.task_id]
                for seg in task.segments:
                    if seg.resource_type in schedule.assigned_resources:
                        resource_id = schedule.assigned_resources[seg.resource_type]
                        # 获取资源单元
                        resource_unit = None
                        for r in scheduler.resources[seg.resource_type]:
                            if r.unit_id == resource_id:
                                resource_unit = r
                                break
                        
                        if resource_unit:
                            # 计算实际执行时间
                            duration = seg.get_duration(resource_unit.bandwidth)
                            resource_busy_time[resource_id] += duration
        
        # 计算利用率百分比
        utilization = {}
        for resource_type, resources in scheduler.resources.items():
            for resource in resources:
                resource_id = resource.unit_id
                if resource_id in resource_busy_time:
                    busy_time = resource_busy_time[resource_id]
                    utilization[resource_id] = (busy_time / time_window) * 100
                else:
                    utilization[resource_id] = 0.0
        
        return utilization
    
    # 替换方法
    scheduler.get_resource_utilization = correct_get_resource_utilization
    
    print("✅ 资源利用率计算已修复")


def patch_elegant_visualization():
    """修补ElegantSchedulerVisualizer以显示正确的利用率"""
    
    try:
        import elegant_visualization
        
        # 保存原始的_add_elegant_metrics方法
        original_add_metrics = elegant_visualization.ElegantSchedulerVisualizer._add_elegant_metrics
        
        def fixed_add_elegant_metrics(self, ax, time_window):
            """修复的度量显示方法"""
            
            # 确保调度器使用正确的利用率计算
            if hasattr(self.scheduler, 'get_resource_utilization'):
                fix_scheduler_utilization_calculation(self.scheduler)
            
            # 调用原始方法
            original_add_metrics(self, ax, time_window)
        
        # 替换方法
        elegant_visualization.ElegantSchedulerVisualizer._add_elegant_metrics = fixed_add_elegant_metrics
        
        print("✅ 可视化利用率显示已修复")
        
    except ImportError:
        print("⚠️  未找到elegant_visualization模块")


if __name__ == "__main__":
    print("资源利用率计算修复模块")
    print("主要功能：")
    print("1. 修复scheduler的get_resource_utilization方法")
    print("2. 正确计算每个资源的实际忙碌时间")
    print("3. 修补可视化模块显示正确的利用率")
