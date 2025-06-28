#!/usr/bin/env python3
"""
利用RuntimeType特性解决T1的segment冲突
通过将T1改为DSP_Runtime来确保绑定执行
"""

from enums import TaskPriority, RuntimeType


def apply_runtime_type_fix(scheduler):
    """应用运行时类型修复"""
    
    print("🔧 应用RuntimeType修复...")
    
    # 修改T1的运行时类型为DSP_Runtime
    if "T1" in scheduler.tasks:
        task1 = scheduler.tasks["T1"]
        print(f"\n修改T1的配置:")
        print(f"  原RuntimeType: {task1.runtime_type.name}")
        print(f"  原Priority: {task1.priority.name}")
        
        # 改为DSP_Runtime以启用绑定执行
        task1.runtime_type = RuntimeType.DSP_RUNTIME
        # 可选：提升优先级确保优先调度
        task1.priority = TaskPriority.HIGH
        
        print(f"  新RuntimeType: {task1.runtime_type.name}")
        print(f"  新Priority: {task1.priority.name}")
        print(f"  ✓ T1现在将使用绑定执行模式")
    
    # 验证调度器是否正确处理DSP_Runtime
    ensure_dsp_runtime_binding(scheduler)
    
    print("✅ RuntimeType修复已应用")


def ensure_dsp_runtime_binding(scheduler):
    """确保调度器正确处理DSP_Runtime的绑定执行"""
    
    print("\n验证DSP_Runtime绑定执行支持:")
    
    # 检查调度器中的相关方法
    if hasattr(scheduler, 'find_bound_resources_with_segmentation'):
        print("  ✓ find_bound_resources_with_segmentation 方法存在")
    else:
        print("  ⚠️ 缺少绑定资源查找方法")
    
    # 检查任务的运行时类型
    dsp_runtime_tasks = []
    acpu_runtime_tasks = []
    
    for task_id, task in scheduler.tasks.items():
        if task.runtime_type == RuntimeType.DSP_RUNTIME:
            dsp_runtime_tasks.append(task_id)
        else:
            acpu_runtime_tasks.append(task_id)
    
    print(f"\nDSP_Runtime任务 (绑定执行): {dsp_runtime_tasks}")
    print(f"ACPU_Runtime任务 (流水线执行): {acpu_runtime_tasks}")
    
    # 分析DSP_Runtime任务的segment结构
    for task_id in dsp_runtime_tasks:
        task = scheduler.tasks[task_id]
        print(f"\n{task_id} segment分析:")
        npu_time = 0
        dsp_time = 0
        
        for i, seg in enumerate(task.segments):
            # 使用40 bandwidth计算
            duration = seg.get_duration(40) if 40 in seg.duration_table else 0
            
            if seg.resource_type.name == "NPU":
                npu_time += duration
            else:
                dsp_time += duration
                
            print(f"  Seg{i}: {seg.resource_type.name} @ {seg.start_time:.1f}ms, duration={duration:.1f}ms")
        
        print(f"  总NPU时间: {npu_time:.1f}ms")
        print(f"  总DSP时间: {dsp_time:.1f}ms")


def debug_scheduling_with_runtime_fix(scheduler):
    """调试运行时修复后的调度"""
    
    print("\n🔍 调试DSP_Runtime绑定执行:")
    
    # 保存原始调度方法
    original_schedule = scheduler.priority_aware_schedule_with_segmentation
    
    def debug_schedule(time_window):
        """带调试信息的调度"""
        
        print("\n开始调度，DSP_Runtime任务将绑定执行...")
        
        # 调用原始调度
        results = original_schedule(time_window)
        
        # 分析T1的调度情况
        t1_schedules = [s for s in scheduler.schedule_history if s.task_id == "T1"]
        
        if t1_schedules:
            print(f"\nT1调度分析 (共{len(t1_schedules)}次):")
            for i, sched in enumerate(t1_schedules[:3]):  # 显示前3次
                print(f"\n  第{i+1}次执行:")
                print(f"    开始时间: {sched.start_time:.1f}ms")
                print(f"    结束时间: {sched.end_time:.1f}ms")
                print(f"    持续时间: {sched.end_time - sched.start_time:.1f}ms")
                print(f"    分配资源: {sched.assigned_resources}")
                
                # 检查是否有其他任务在T1执行期间插入
                conflicts = []
                for other in scheduler.schedule_history:
                    if other.task_id != "T1":
                        # 检查时间重叠
                        if not (other.end_time <= sched.start_time or 
                               other.start_time >= sched.end_time):
                            # 检查资源重叠
                            for res_type, res_id in other.assigned_resources.items():
                                if res_id in sched.assigned_resources.values():
                                    conflicts.append(f"{other.task_id} 使用 {res_id}")
                
                if conflicts:
                    print(f"    ⚠️ 发现冲突: {conflicts}")
                else:
                    print(f"    ✅ 绑定执行成功，无任务打断")
        
        return results
    
    # 临时替换调度方法以添加调试
    scheduler.priority_aware_schedule_with_segmentation = debug_schedule
    
    print("  ✓ 调试模式已启用")


def simple_priority_adjustment(scheduler):
    """简单的优先级调整方案"""
    
    print("\n📊 优先级调整建议:")
    
    # 分析当前优先级分布
    priority_count = {}
    for task_id, task in scheduler.tasks.items():
        priority = task.priority.name
        if priority not in priority_count:
            priority_count[priority] = []
        priority_count[priority].append(task_id)
    
    print("\n当前优先级分布:")
    for priority in ["CRITICAL", "HIGH", "NORMAL", "LOW"]:
        if priority in priority_count:
            print(f"  {priority}: {priority_count[priority]}")
    
    # 建议调整
    print("\n建议的优先级调整:")
    print("  1. T1改为DSP_Runtime + HIGH优先级 (避免segment冲突)")
    print("  2. T8已经是DSP_Runtime，保持不变")
    print("  3. 其他任务保持ACPU_Runtime以支持流水线执行")
    
    # 可选：调整其他任务优先级
    if "T6" in scheduler.tasks and scheduler.tasks["T6"].fps_requirement > 50:
        print("  4. T6的FPS要求很高(100fps)，但可以降低优先级因为执行时间短")


if __name__ == "__main__":
    print("RuntimeType修复方案")
    print("=" * 60)
    print("利用DSP_Runtime的绑定执行特性解决T1的segment冲突")
    print("\n关键点：")
    print("1. DSP_Runtime会自动绑定任务的所有segments")
    print("2. 不会被其他任务打断")
    print("3. 无需修改任务的耗时模型")
