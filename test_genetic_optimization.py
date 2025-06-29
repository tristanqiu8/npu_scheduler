#!/usr/bin/env python3
"""
测试遗传算法优化器
对比基线结果和优化结果
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scheduler import MultiResourceScheduler
from real_task import create_real_tasks
from modular_scheduler_fixes import apply_basic_fixes
from genetic_task_optimizer import run_genetic_optimization, GeneticTaskOptimizer
from elegant_visualization import ElegantSchedulerVisualizer
from fixed_validation_and_metrics import validate_schedule_correctly
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免Windows GUI问题
import matplotlib.pyplot as plt
# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def create_test_system():
    """创建测试系统"""
    
    print("🔧 创建测试系统...")
    
    # 创建调度器
    scheduler = MultiResourceScheduler(enable_segmentation=True)
    
    # 添加资源 - 2个NPU和2个DSP
    scheduler.add_npu("NPU_0", bandwidth=40.0)
    scheduler.add_dsp("DSP_0", bandwidth=40.0)
    
    return scheduler


def run_baseline_test(scheduler, tasks, time_window):
    """运行基线测试"""
    
    print("\n" + "=" * 80)
    print("📊 运行基线测试（原始配置）")
    print("=" * 80)
    
    # 清空调度历史
    scheduler.schedule_history.clear()
    
    # 运行调度
    results = scheduler.priority_aware_schedule_with_segmentation(time_window)
    print(f"\n基线调度完成: {len(results)} 个事件")
    
    # 验证
    is_valid, conflicts = validate_schedule_correctly(scheduler)
    
    # 分析FPS
    baseline_stats = analyze_fps_satisfaction(scheduler, time_window)
    
    return baseline_stats, len(conflicts)


def run_genetic_optimization_test(scheduler, tasks, time_window):
    """运行遗传算法优化测试"""
    
    print("\n" + "=" * 80)
    print("🧬 运行遗传算法优化")
    print("=" * 80)
    
    # 创建优化器并运行
    optimizer = GeneticTaskOptimizer(scheduler, tasks, time_window)
    
    # 调整参数以加快测试
    optimizer.population_size = 30
    optimizer.generations = 50
    optimizer.elite_size = 5
    
    # 运行优化
    best_individual = optimizer.optimize()
    optimizer.print_optimization_report()
    
    # 重新运行调度以获取最终结果
    scheduler.schedule_history.clear()
    results = scheduler.priority_aware_schedule_with_segmentation(time_window)
    print(f"\n优化后调度完成: {len(results)} 个事件")
    
    # 验证
    is_valid, conflicts = validate_schedule_correctly(scheduler)
    
    # 分析FPS
    optimized_stats = analyze_fps_satisfaction(scheduler, time_window)
    
    return optimized_stats, len(conflicts), optimizer


def analyze_fps_satisfaction(scheduler, time_window):
    """分析FPS满足情况"""
    
    stats = {
        'task_fps': {},
        'satisfied_count': 0,
        'total_fps_rate': 0.0,
        'resource_utilization': {}
    }
    
    # 统计任务执行次数
    task_counts = defaultdict(int)
    for event in scheduler.schedule_history:
        task_counts[event.task_id] += 1
    
    # 计算FPS满足率
    for task in scheduler.tasks.values():
        count = task_counts[task.task_id]
        expected = int((time_window / 1000.0) * task.fps_requirement)
        
        if expected > 0:
            fps_rate = min(1.0, count / expected)
            stats['task_fps'][task.task_id] = {
                'name': task.name,
                'count': count,
                'expected': expected,
                'fps_rate': fps_rate,
                'satisfied': fps_rate >= 0.95
            }
            
            stats['total_fps_rate'] += fps_rate
            if fps_rate >= 0.95:
                stats['satisfied_count'] += 1
    
    # 计算资源利用率
    for res_type, resources in scheduler.resources.items():
        # 检查resources是否是字典
        if isinstance(resources, dict):
            resource_items = resources.items()
        elif isinstance(resources, list):
            # 如果是列表，创建索引作为键
            resource_items = [(f"{res_type.value}_{i}", res) for i, res in enumerate(resources)]
        else:
            continue
            
        for res_id, resource in resource_items:
            busy_time = 0.0
            for event in scheduler.schedule_history:
                if event.assigned_resources.get(res_type) == res_id:
                    busy_time += event.end_time - event.start_time
            
            utilization = busy_time / time_window if time_window > 0 else 0
            stats['resource_utilization'][res_id] = utilization
    
    return stats


def print_comparison_report(baseline_stats, optimized_stats, baseline_conflicts, optimized_conflicts):
    """打印对比报告"""
    
    print("\n" + "=" * 80)
    print("📊 优化效果对比报告")
    print("=" * 80)
    
    # 整体指标对比
    print("\n🎯 整体指标:")
    print(f"{'指标':<20} {'基线':<15} {'优化后':<15} {'改进':<15}")
    print("-" * 65)
    
    baseline_satisfied = baseline_stats['satisfied_count']
    optimized_satisfied = optimized_stats['satisfied_count']
    total_tasks = len(baseline_stats['task_fps'])
    
    print(f"{'资源冲突数':<20} {baseline_conflicts:<15} {optimized_conflicts:<15} "
          f"{baseline_conflicts - optimized_conflicts:<15}")
    
    print(f"{'FPS满足任务数':<20} {baseline_satisfied}/{total_tasks:<14} "
          f"{optimized_satisfied}/{total_tasks:<14} "
          f"{optimized_satisfied - baseline_satisfied:<15}")
    
    baseline_avg_fps = baseline_stats['total_fps_rate'] / total_tasks
    optimized_avg_fps = optimized_stats['total_fps_rate'] / total_tasks
    
    print(f"{'平均FPS满足率':<20} {baseline_avg_fps:.1%}{'':12} "
          f"{optimized_avg_fps:.1%}{'':12} "
          f"{(optimized_avg_fps - baseline_avg_fps):.1%}")
    
    baseline_avg_util = sum(baseline_stats['resource_utilization'].values()) / len(baseline_stats['resource_utilization'])
    optimized_avg_util = sum(optimized_stats['resource_utilization'].values()) / len(optimized_stats['resource_utilization'])
    
    print(f"{'平均资源利用率':<20} {baseline_avg_util:.1%}{'':12} "
          f"{optimized_avg_util:.1%}{'':12} "
          f"{(optimized_avg_util - baseline_avg_util):.1%}")
    
    # 任务级别对比
    print("\n📋 任务级别FPS满足情况:")
    print(f"{'任务ID':<8} {'任务名称':<15} {'基线FPS':<15} {'优化后FPS':<15} {'状态':<10}")
    print("-" * 70)
    
    for task_id in sorted(baseline_stats['task_fps'].keys()):
        baseline_task = baseline_stats['task_fps'][task_id]
        optimized_task = optimized_stats['task_fps'][task_id]
        
        baseline_fps_str = f"{baseline_task['count']}/{baseline_task['expected']} ({baseline_task['fps_rate']:.1%})"
        optimized_fps_str = f"{optimized_task['count']}/{optimized_task['expected']} ({optimized_task['fps_rate']:.1%})"
        
        status = "✅ 改进" if optimized_task['fps_rate'] > baseline_task['fps_rate'] else \
                 "⚠️ 不变" if optimized_task['fps_rate'] == baseline_task['fps_rate'] else "❌ 下降"
        
        print(f"{task_id:<8} {baseline_task['name']:<15} {baseline_fps_str:<15} "
              f"{optimized_fps_str:<15} {status:<10}")
    
    # 资源利用率对比
    print("\n💻 资源利用率:")
    print(f"{'资源ID':<10} {'基线利用率':<15} {'优化后利用率':<15}")
    print("-" * 40)
    
    for res_id in sorted(baseline_stats['resource_utilization'].keys()):
        baseline_util = baseline_stats['resource_utilization'][res_id]
        optimized_util = optimized_stats['resource_utilization'][res_id]
        
        print(f"{res_id:<10} {baseline_util:<15.1%} {optimized_util:<15.1%}")


def plot_evolution_curve(optimizer: GeneticTaskOptimizer):
    """绘制进化曲线"""
    
    if not optimizer.generation_history:
        return
    
    try:
        generations = [h['generation'] for h in optimizer.generation_history]
        best_fitness = [h['best_fitness'] for h in optimizer.generation_history]
        avg_fitness = [h['avg_fitness'] for h in optimizer.generation_history]
        fps_rates = [h['best_fps_rate'] for h in optimizer.generation_history]
        conflicts = [h['best_conflicts'] for h in optimizer.generation_history]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # 适应度曲线
        ax1.plot(generations, best_fitness, 'b-', label='Best Fitness')
        ax1.plot(generations, avg_fitness, 'r--', label='Average Fitness')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness')
        ax1.set_title('Fitness Evolution')
        ax1.legend()
        ax1.grid(True)
        
        # FPS满足率曲线
        ax2.plot(generations, fps_rates, 'g-')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('FPS Satisfaction Rate')
        ax2.set_title('FPS Satisfaction Evolution')
        ax2.set_ylim(0, 1.1)
        ax2.grid(True)
        
        # 资源冲突数曲线
        ax3.plot(generations, conflicts, 'r-')
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Conflict Count')
        ax3.set_title('Resource Conflicts Evolution')
        ax3.grid(True)
        
        # 改进率曲线
        if len(best_fitness) > 1:
            improvement_rates = [(best_fitness[i] - best_fitness[0]) / abs(best_fitness[0]) * 100 
                               for i in range(len(best_fitness))]
            ax4.plot(generations, improvement_rates, 'm-')
            ax4.set_xlabel('Generation')
            ax4.set_ylabel('Improvement Rate (%)')
            ax4.set_title('Fitness Improvement Rate')
            ax4.grid(True)
        
        plt.tight_layout()
        
        # 保存图片，处理Windows路径
        output_path = os.path.join(os.getcwd(), 'genetic_evolution_curves.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()  # 关闭图形，释放内存
        
        print(f"\n✅ 进化曲线已保存到 {output_path}")
        
    except Exception as e:
        print(f"\n⚠️ 进化曲线生成失败: {e}")
        import traceback
        traceback.print_exc()


def save_optimization_report(baseline_stats, optimized_stats, baseline_conflicts, 
                           optimized_conflicts, optimizer):
    """保存详细的优化报告到文件"""
    
    try:
        report_path = os.path.join(os.getcwd(), 'genetic_optimization_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("遗传算法优化报告\n")
            f.write("=" * 80 + "\n\n")
            
            # 写入配置信息
            f.write("优化器配置:\n")
            f.write(f"  - 种群大小: {optimizer.population_size}\n")
            f.write(f"  - 精英个体: {optimizer.elite_size}\n")
            f.write(f"  - 变异率: {optimizer.mutation_rate}\n")
            f.write(f"  - 交叉率: {optimizer.crossover_rate}\n")
            f.write(f"  - 进化代数: {len(optimizer.generation_history)}\n\n")
            
            # 写入整体改进
            f.write("整体性能改进:\n")
            f.write(f"  - 资源冲突: {baseline_conflicts} → {optimized_conflicts} "
                   f"(减少 {baseline_conflicts - optimized_conflicts})\n")
            
            baseline_avg_fps = baseline_stats['total_fps_rate'] / len(baseline_stats['task_fps'])
            optimized_avg_fps = optimized_stats['total_fps_rate'] / len(optimized_stats['task_fps'])
            f.write(f"  - 平均FPS满足率: {baseline_avg_fps:.1%} → {optimized_avg_fps:.1%} "
                   f"(改进 {(optimized_avg_fps - baseline_avg_fps):.1%})\n\n")
            
            # 写入任务级别详情
            f.write("任务配置变化:\n")
            f.write("-" * 70 + "\n")
            
            for task_id in sorted(baseline_stats['task_fps'].keys()):
                if hasattr(optimizer, 'best_individual'):
                    orig_priority = optimizer.original_config.task_priorities.get(task_id)
                    new_priority = optimizer.best_individual.task_priorities.get(task_id)
                    orig_runtime = optimizer.original_config.task_runtime_types.get(task_id)
                    new_runtime = optimizer.best_individual.task_runtime_types.get(task_id)
                    
                    f.write(f"\n{task_id} ({baseline_stats['task_fps'][task_id]['name']}):\n")
                    if orig_priority != new_priority:
                        f.write(f"  - 优先级: {orig_priority.name} → {new_priority.name}\n")
                    if orig_runtime != new_runtime:
                        f.write(f"  - 运行时: {orig_runtime.value} → {new_runtime.value}\n")
                    
                    baseline_fps = baseline_stats['task_fps'][task_id]['fps_rate']
                    optimized_fps = optimized_stats['task_fps'][task_id]['fps_rate']
                    f.write(f"  - FPS满足率: {baseline_fps:.1%} → {optimized_fps:.1%}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("报告生成时间: " + str(os.path.getmtime(report_path)) + "\n")
        
        print(f"\n✅ 详细报告已保存到 {report_path}")
        
    except Exception as e:
        print(f"\n⚠️ 报告保存失败: {e}")


def export_optimization_config(optimizer: GeneticTaskOptimizer):
    """导出最优配置为可重用的Python代码"""
    
    try:
        config_path = os.path.join(os.getcwd(), 'optimal_config.py')
        
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write("#!/usr/bin/env python3\n")
            f.write('"""\n')
            f.write("遗传算法找到的最优配置\n")
            f.write("可直接导入使用\n")
            f.write('"""\n\n')
            f.write("from enums import TaskPriority, RuntimeType, SegmentationStrategy\n\n")
            
            f.write("# 最优任务配置\n")
            f.write("OPTIMAL_CONFIG = {\n")
            
            if hasattr(optimizer, 'best_individual'):
                for task_id in sorted(optimizer.best_individual.task_priorities.keys()):
                    f.write(f"    '{task_id}': {{\n")
                    f.write(f"        'priority': TaskPriority.{optimizer.best_individual.task_priorities[task_id].name},\n")
                    f.write(f"        'runtime_type': RuntimeType.{optimizer.best_individual.task_runtime_types[task_id].name},\n")
                    f.write(f"        'segmentation_strategy': SegmentationStrategy.{optimizer.best_individual.task_segmentation_strategies[task_id].name},\n")
                    f.write("    },\n")
            
            f.write("}\n\n")
            
            f.write("def apply_optimal_config(scheduler, tasks):\n")
            f.write('    """应用最优配置到任务"""\n')
            f.write("    for task in tasks:\n")
            f.write("        if task.task_id in OPTIMAL_CONFIG:\n")
            f.write("            config = OPTIMAL_CONFIG[task.task_id]\n")
            f.write("            task.priority = config['priority']\n")
            f.write("            task.runtime_type = config['runtime_type']\n")
            f.write("            task.segmentation_strategy = config['segmentation_strategy']\n")
        
        print(f"\n✅ 最优配置已导出到 {config_path}")
        
    except Exception as e:
        print(f"\n⚠️ 配置导出失败: {e}")


def analyze_task_interactions(scheduler, optimizer):
    """分析任务间的相互影响"""
    
    print("\n📊 任务交互分析:")
    print("=" * 60)
    
    # 分析依赖关系
    dependency_map = {}
    for task in scheduler.tasks.values():
        if task.dependencies:
            dependency_map[task.task_id] = list(task.dependencies)
    
    if dependency_map:
        print("\n依赖关系:")
        for task_id, deps in dependency_map.items():
            print(f"  {task_id} 依赖于: {', '.join(deps)}")
    else:
        print("\n无任务依赖关系")
    
    # 分析资源竞争
    print("\n资源竞争分析:")
    resource_usage = defaultdict(list)
    
    for event in scheduler.schedule_history:
        for res_type, res_id in event.assigned_resources.items():
            resource_usage[res_id].append((event.task_id, event.start_time, event.end_time))
    
    for res_id, usages in resource_usage.items():
        if len(usages) > 1:
            print(f"\n  {res_id} 被以下任务使用:")
            for task_id, start, end in sorted(usages, key=lambda x: x[1])[:5]:
                print(f"    - {task_id}: {start:.1f}ms - {end:.1f}ms")


def main():
    """主测试函数"""
    
    print("=" * 80)
    print("🧬 遗传算法任务优化完整测试")
    print("=" * 80)
    
    # 1. 创建系统
    scheduler = create_test_system()
    
    # 2. 应用基础修复
    fix_manager = apply_basic_fixes(scheduler)
    
    # 3. 创建任务
    tasks = create_real_tasks()
    for task in tasks:
        scheduler.add_task(task)
    
    print(f"\n✅ 添加了 {len(tasks)} 个任务")
    
    # 4. 设置时间窗口
    time_window = 200.0
    print(f"\n⏱️ 时间窗口: {time_window}ms")
    
    # 5. 运行基线测试
    baseline_stats, baseline_conflicts = run_baseline_test(scheduler, tasks, time_window)
    
    # 生成基线可视化
    try:
        viz = ElegantSchedulerVisualizer(scheduler)
        # 尝试不同的方法名和参数
        try:
            viz.plot_elegant_gantt()  # 尝试无参数
            plt.savefig('baseline_schedule.png')
            plt.close()
        except:
            try:
                viz.plot_gantt_chart()  # 尝试其他方法名
                plt.savefig('baseline_schedule.png')
                plt.close()
            except:
                pass
        
        viz.export_chrome_tracing("baseline_trace.json")
        print("\n✅ 基线可视化已生成")
    except Exception as e:
        print(f"\n⚠️ 基线可视化生成失败: {e}")
        # 继续执行，不中断程序
    
    # 6. 运行遗传算法优化
    optimized_stats, optimized_conflicts, optimizer = run_genetic_optimization_test(
        scheduler, tasks, time_window
    )
    
    # 生成优化后的可视化
    try:
        viz = ElegantSchedulerVisualizer(scheduler)
        # 尝试不同的方法名和参数
        try:
            viz.plot_elegant_gantt()  # 尝试无参数
            plt.savefig('genetic_optimized_schedule.png')
            plt.close()
        except:
            try:
                viz.plot_gantt_chart()  # 尝试其他方法名
                plt.savefig('genetic_optimized_schedule.png')
                plt.close()
            except:
                pass
        
        viz.export_chrome_tracing("genetic_optimized_trace.json")
        print("\n✅ 优化后可视化已生成")
    except Exception as e:
        print(f"\n⚠️ 优化后可视化生成失败: {e}")
        # 继续执行，不中断程序
    
    # 7. 生成进化曲线
    plot_evolution_curve(optimizer)
    
    # 8. 打印对比报告
    print_comparison_report(baseline_stats, optimized_stats, baseline_conflicts, optimized_conflicts)
    
    # 9. 分析任务交互
    analyze_task_interactions(scheduler, optimizer)
    
    # 10. 保存详细报告
    save_optimization_report(baseline_stats, optimized_stats, baseline_conflicts, 
                           optimized_conflicts, optimizer)
    
    # 11. 导出最优配置
    export_optimization_config(optimizer)
    
    # 12. 总结
    print("\n" + "=" * 80)
    print("🎯 测试总结")
    print("=" * 80)
    
    improvement = ((optimized_stats['satisfied_count'] - baseline_stats['satisfied_count']) / 
                   len(tasks) * 100)
    
    print(f"\n✅ 遗传算法优化完成")
    print(f"  - FPS满足任务改进: {improvement:.1f}%")
    print(f"  - 资源冲突减少: {baseline_conflicts - optimized_conflicts}")
    print(f"  - 进化代数: {len(optimizer.generation_history)}")
    
    print("\n📁 生成的文件:")
    print("  - baseline_schedule.png: 基线调度甘特图")
    print("  - genetic_optimized_schedule.png: 优化后调度甘特图")
    print("  - genetic_evolution_curves.png: 进化过程曲线")
    print("  - baseline_trace.json: 基线Chrome追踪文件")
    print("  - genetic_optimized_trace.json: 优化后Chrome追踪文件")
    print("  - genetic_optimization_report.txt: 详细优化报告")
    print("  - optimal_config.py: 最优配置文件")
    
    print("\n💡 建议:")
    print("  1. 查看甘特图对比调度效果")
    print("  2. 分析进化曲线了解优化过程")
    print("  3. 使用Chrome追踪文件进行详细分析")
    print("  4. 根据任务特性调整遗传算法参数")
    print("  5. 使用optimal_config.py快速应用最优配置")


if __name__ == "__main__":
    main()
