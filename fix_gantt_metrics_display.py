#!/usr/bin/env python3
"""
修复甘特图中的度量显示
1. 分别显示每个资源的利用率
2. 正确计算分段任务数量
"""

def patch_gantt_metrics_display():
    """修补甘特图的度量显示"""
    
    try:
        import elegant_visualization
        from matplotlib import patches
        
        # 保存原始方法
        original_add_metrics = elegant_visualization.ElegantSchedulerVisualizer._add_elegant_metrics
        
        def improved_add_elegant_metrics(self, ax, time_window):
            """改进的度量显示方法"""
            
            # 获取资源利用率
            utilization = self.scheduler.get_resource_utilization(time_window)
            
            # 计算分段任务数量（修复）
            segmented_count = 0
            for task in self.scheduler.tasks.values():
                # 检查任务是否真的被分段了
                if hasattr(task, 'is_segmented') and task.is_segmented:
                    segmented_count += 1
                elif hasattr(task, 'current_segmentation'):
                    # 检查是否有实际的分段
                    for seg_id, cuts in task.current_segmentation.items():
                        if cuts and len(cuts) > 0:
                            segmented_count += 1
                            break
            
            # 计算总开销
            total_overhead = 0.0
            for schedule in self.scheduler.schedule_history:
                if hasattr(schedule, 'segmentation_overhead'):
                    total_overhead += schedule.segmentation_overhead
            
            # 完成时间
            complete_time = self.scheduler.schedule_history[-1].end_time if self.scheduler.schedule_history else 0
            
            # 构建详细的利用率字符串
            util_parts = []
            
            # NPU利用率
            npu_utils = []
            for res_id, util in sorted(utilization.items()):
                if 'NPU' in res_id:
                    npu_utils.append(f"{res_id}: {util:.0f}%")
            if npu_utils:
                util_parts.append("NPU(" + ", ".join(npu_utils) + ")")
            
            # DSP利用率
            dsp_utils = []
            for res_id, util in sorted(utilization.items()):
                if 'DSP' in res_id:
                    dsp_utils.append(f"{res_id}: {util:.0f}%")
            if dsp_utils:
                util_parts.append("DSP(" + ", ".join(dsp_utils) + ")")
            
            # 构建完整的度量字符串
            metrics = (f"Utilization: {' | '.join(util_parts)} | "
                      f"Segmented: {segmented_count} tasks | "
                      f"Overhead: {total_overhead:.0f}ms | "
                      f"Complete: {complete_time:.0f}ms")
            
            # 显示度量（可能需要更大的空间）
            ax.text(0.02, 0.02, metrics, 
                   transform=ax.transAxes, 
                   fontsize=8,  # 稍微减小字体
                   color='#6B7280',
                   bbox=dict(boxstyle='round,pad=0.3', 
                            facecolor='white', 
                            edgecolor='#E5E7EB',
                            linewidth=0.5,
                            alpha=0.95))
        
        # 替换方法
        elegant_visualization.ElegantSchedulerVisualizer._add_elegant_metrics = improved_add_elegant_metrics
        
        print("✅ 甘特图度量显示已改进")
        
    except ImportError:
        print("⚠️  未找到elegant_visualization模块")


def fix_segmentation_stats(scheduler):
    """修复分段统计信息"""
    
    print("🔧 修复分段统计...")
    
    # 重新计算分段统计
    segmented_tasks = 0
    total_overhead = 0.0
    
    # 检查每个任务
    for task in scheduler.tasks.values():
        is_actually_segmented = False
        
        # 方法1：检查is_segmented标志
        if hasattr(task, 'is_segmented') and task.is_segmented:
            is_actually_segmented = True
        
        # 方法2：检查current_segmentation
        elif hasattr(task, 'current_segmentation'):
            for seg_id, cuts in task.current_segmentation.items():
                if cuts and len(cuts) > 0:
                    is_actually_segmented = True
                    break
        
        # 方法3：检查分段策略
        elif hasattr(task, 'segmentation_strategy'):
            strategy_name = task.segmentation_strategy.name if hasattr(task.segmentation_strategy, 'name') else str(task.segmentation_strategy)
            if strategy_name not in ["NO_SEGMENTATION", "NONE"]:
                # 检查是否有实际的切点
                for seg in task.segments:
                    if hasattr(seg, 'cut_points') and seg.cut_points:
                        is_actually_segmented = True
                        break
        
        if is_actually_segmented:
            segmented_tasks += 1
            
            # 计算开销
            if hasattr(task, 'total_segmentation_overhead'):
                total_overhead += task.total_segmentation_overhead
    
    # 更新调度器的统计信息
    scheduler.segmentation_stats['segmented_tasks'] = segmented_tasks
    scheduler.segmentation_stats['total_overhead'] = total_overhead
    
    # 从调度历史中计算实际开销
    actual_overhead = 0.0
    for schedule in scheduler.schedule_history:
        if hasattr(schedule, 'segmentation_overhead'):
            actual_overhead += schedule.segmentation_overhead
    
    if actual_overhead > 0:
        scheduler.segmentation_stats['total_overhead'] = actual_overhead
    
    print(f"  ✓ 分段任务数: {segmented_tasks}")
    print(f"  ✓ 总开销: {scheduler.segmentation_stats['total_overhead']:.1f}ms")
    
    print("✅ 分段统计已修复")


if __name__ == "__main__":
    print("甘特图度量显示修复")
    print("功能：")
    print("1. 分别显示每个资源的利用率")
    print("2. 正确计算分段任务数量")
    print("3. 修复分段统计信息")
