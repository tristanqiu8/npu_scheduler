from enum import Enum
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
import heapq
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class ResourceType(Enum):
    """Resource type enumeration"""
    DSP = "DSP"
    NPU = "NPU"

class TaskPriority(Enum):
    """Task priority levels (from high to low)"""
    CRITICAL = 0     # Highest priority - safety critical tasks
    HIGH = 1         # High priority - real-time tasks
    NORMAL = 2       # Normal priority - regular tasks
    LOW = 3          # Low priority - background tasks
    
    def __lt__(self, other):
        """Enable priority comparison"""
        return self.value < other.value

@dataclass
class ResourceSegment:
    """Resource usage segment"""
    resource_type: ResourceType
    duration_table: Dict[float, float]  # {BW: duration} lookup table
    start_time: float  # Start time relative to task beginning
    
    def get_duration(self, bw: float) -> float:
        """Get execution time based on bandwidth"""
        if bw in self.duration_table:
            return self.duration_table[bw]
        closest_bw = min(self.duration_table.keys(), key=lambda x: abs(x - bw))
        return self.duration_table[closest_bw]

@dataclass
class ResourceUnit:
    """Resource unit (NPU or DSP instance)"""
    unit_id: str
    resource_type: ResourceType
    bandwidth: float  # Bandwidth capability of this unit
    
    def __hash__(self):
        return hash(self.unit_id)

@dataclass
class TaskScheduleInfo:
    """Task scheduling information"""
    task_id: str
    start_time: float
    end_time: float
    assigned_resources: Dict[ResourceType, str]  # {resource type: assigned resource ID}
    actual_latency: float  # Actual latency from scheduling to completion

class NNTask:
    """Neural network task class"""
    
    def __init__(self, task_id: str, name: str = "", priority: TaskPriority = TaskPriority.NORMAL):
        self.task_id = task_id
        self.name = name or f"Task_{task_id}"
        self.priority = priority  # Task priority level
        self.segments: List[ResourceSegment] = []
        self.dependencies: Set[str] = set()
        self.fps_requirement: float = 30.0
        self.latency_requirement: float = 100.0
        
        # Scheduling related info
        self.schedule_info: Optional[TaskScheduleInfo] = None
        self.last_execution_time: float = -float('inf')
        self.ready_time: float = 0  # Time when task becomes ready (dependencies satisfied)
    
    def set_npu_only(self, duration_table: Dict[float, float]):
        """Set as NPU-only task"""
        self.segments = [ResourceSegment(ResourceType.NPU, duration_table, 0)]
        
    def set_dsp_npu_sequence(self, segments: List[Tuple[ResourceType, Dict[float, float], float]]):
        """Set as DSP+NPU mixed execution task"""
        self.segments = []
        for resource_type, duration_table, start_time in segments:
            self.segments.append(ResourceSegment(resource_type, duration_table, start_time))
    
    def add_dependency(self, task_id: str):
        self.dependencies.add(task_id)
    
    def add_dependencies(self, task_ids: List[str]):
        self.dependencies.update(task_ids)
    
    def set_performance_requirements(self, fps: float, latency: float):
        self.fps_requirement = fps
        self.latency_requirement = latency
    
    def get_total_duration(self, resource_bw_map: Dict[ResourceType, float]) -> float:
        """Get total execution time based on assigned resource bandwidths"""
        if not self.segments:
            return 0
        max_end_time = 0
        for seg in self.segments:
            bw = resource_bw_map.get(seg.resource_type, 1.0)
            end_time = seg.start_time + seg.get_duration(bw)
            max_end_time = max(max_end_time, end_time)
        return max_end_time
    
    @property
    def min_interval_ms(self) -> float:
        """Minimum scheduling interval based on FPS requirement"""
        return 1000.0 / self.fps_requirement if self.fps_requirement > 0 else float('inf')
    
    @property
    def uses_dsp(self) -> bool:
        return any(seg.resource_type == ResourceType.DSP for seg in self.segments)
    
    @property
    def uses_npu(self) -> bool:
        return any(seg.resource_type == ResourceType.NPU for seg in self.segments)
    
    def __repr__(self):
        sched_str = f", scheduled@{self.schedule_info.start_time:.1f}ms" if self.schedule_info else ""
        return f"Task{self.task_id}({self.name}, priority={self.priority.name}, fps={self.fps_requirement}{sched_str})"


class ResourcePriorityQueues:
    """Priority queues for a single resource"""
    
    def __init__(self, resource_id: str):
        self.resource_id = resource_id
        # Create a queue for each priority level
        self.queues = {
            priority: deque() for priority in TaskPriority
        }
        self.available_time = 0.0  # When resource becomes available
    
    def add_task(self, task: NNTask, ready_time: float):
        """Add task to appropriate priority queue"""
        self.queues[task.priority].append((task, ready_time))
    
    def get_next_task(self, current_time: float) -> Optional[Tuple[NNTask, float]]:
        """Get next task to execute based on priority"""
        # Check queues from highest to lowest priority
        for priority in TaskPriority:
            queue = self.queues[priority]
            # Remove tasks that are not ready yet
            ready_tasks = []
            while queue:
                task, ready_time = queue.popleft()
                if ready_time <= current_time and task.last_execution_time + task.min_interval_ms <= current_time:
                    ready_tasks.append((task, ready_time))
                else:
                    # Put back if not ready
                    queue.append((task, ready_time))
                    break
            
            # If we found ready tasks at this priority level, return the first one
            if ready_tasks:
                # Put remaining ready tasks back
                for t in ready_tasks[1:]:
                    queue.append(t)
                return ready_tasks[0]
        
        return None
    
    def has_higher_priority_tasks(self, priority: TaskPriority, current_time: float) -> bool:
        """Check if there are higher priority tasks waiting"""
        for p in TaskPriority:
            if p < priority:  # Higher priority
                # Check if any task in this queue is ready
                for task, ready_time in self.queues[p]:
                    if (ready_time <= current_time and 
                        task.last_execution_time + task.min_interval_ms <= current_time):
                        return True
        return False


class MultiResourceScheduler:
    """Multi-resource scheduler with priority queue support"""
    
    def __init__(self):
        self.tasks: Dict[str, NNTask] = {}
        self.resources: Dict[ResourceType, List[ResourceUnit]] = {
            ResourceType.NPU: [],
            ResourceType.DSP: []
        }
        # Priority queues for each resource
        self.resource_queues: Dict[str, ResourcePriorityQueues] = {}
        self.schedule_history: List[TaskScheduleInfo] = []
        
    def add_npu(self, npu_id: str, bandwidth: float):
        """Add NPU resource"""
        npu = ResourceUnit(npu_id, ResourceType.NPU, bandwidth)
        self.resources[ResourceType.NPU].append(npu)
        self.resource_queues[npu_id] = ResourcePriorityQueues(npu_id)
        
    def add_dsp(self, dsp_id: str, bandwidth: float):
        """Add DSP resource"""
        dsp = ResourceUnit(dsp_id, ResourceType.DSP, bandwidth)
        self.resources[ResourceType.DSP].append(dsp)
        self.resource_queues[dsp_id] = ResourcePriorityQueues(dsp_id)
    
    def add_task(self, task: NNTask):
        """Add task"""
        self.tasks[task.task_id] = task
    
    def priority_aware_schedule(self, time_window: float = 1000.0) -> List[TaskScheduleInfo]:
        """Priority-aware scheduling algorithm
        
        Key features:
        - Each resource maintains separate queues for each priority level
        - Higher priority tasks must complete before lower priority tasks
        - Tasks are distributed to resources that can execute them
        """
        # Reset scheduling state
        for queue in self.resource_queues.values():
            queue.available_time = 0.0
            for p in TaskPriority:
                queue.queues[p].clear()
        
        for task in self.tasks.values():
            task.schedule_info = None
            task.last_execution_time = -float('inf')
            task.ready_time = 0
        
        self.schedule_history.clear()
        
        # Track task execution counts for FPS calculation
        task_execution_count = defaultdict(int)
        current_time = 0.0
        
        while current_time < time_window:
            # Phase 1: Check which tasks are ready (dependencies satisfied)
            for task in self.tasks.values():
                # Skip if task was recently executed
                if task.last_execution_time + task.min_interval_ms > current_time:
                    continue
                
                # Check dependencies
                deps_satisfied = True
                max_dep_end_time = 0.0
                
                for dep_id in task.dependencies:
                    dep_task = self.tasks.get(dep_id)
                    if dep_task:
                        if task_execution_count[dep_id] <= task_execution_count[task.task_id]:
                            deps_satisfied = False
                            break
                        if dep_task.schedule_info:
                            max_dep_end_time = max(max_dep_end_time, dep_task.schedule_info.end_time)
                
                if deps_satisfied:
                    task.ready_time = max(current_time, max_dep_end_time)
            
            # Phase 2: Distribute ready tasks to appropriate resource queues
            scheduled_any = False
            
            for task in self.tasks.values():
                if (task.ready_time <= current_time and 
                    task.last_execution_time + task.min_interval_ms <= current_time):
                    
                    # Try to schedule this task
                    assigned_resources = {}
                    can_schedule = True
                    earliest_start = current_time
                    
                    # Check resource availability for all segments
                    for seg in task.segments:
                        best_resource = None
                        best_start_time = float('inf')
                        
                        # Find best resource for this segment
                        for resource in self.resources[seg.resource_type]:
                            queue = self.resource_queues[resource.unit_id]
                            
                            # Check if higher priority tasks are waiting
                            if queue.has_higher_priority_tasks(task.priority, current_time):
                                continue
                            
                            # Calculate when this resource could start this task
                            resource_start = max(queue.available_time, earliest_start + seg.start_time)
                            
                            if resource_start < best_start_time:
                                best_start_time = resource_start
                                best_resource = resource
                        
                        if best_resource:
                            assigned_resources[seg.resource_type] = best_resource.unit_id
                            earliest_start = max(earliest_start, best_start_time - seg.start_time)
                        else:
                            can_schedule = False
                            break
                    
                    if can_schedule:
                        # Execute the scheduling
                        actual_start = earliest_start
                        actual_end = actual_start
                        
                        for seg in task.segments:
                            resource_id = assigned_resources[seg.resource_type]
                            resource = next(r for r in self.resources[seg.resource_type] 
                                          if r.unit_id == resource_id)
                            
                            seg_start = actual_start + seg.start_time
                            seg_duration = seg.get_duration(resource.bandwidth)
                            seg_end = seg_start + seg_duration
                            
                            # Update resource availability
                            self.resource_queues[resource_id].available_time = seg_end
                            actual_end = max(actual_end, seg_end)
                        
                        # Create schedule info
                        schedule_info = TaskScheduleInfo(
                            task_id=task.task_id,
                            start_time=actual_start,
                            end_time=actual_end,
                            assigned_resources=assigned_resources,
                            actual_latency=actual_end - current_time
                        )
                        
                        task.schedule_info = schedule_info
                        task.last_execution_time = actual_start
                        self.schedule_history.append(schedule_info)
                        task_execution_count[task.task_id] += 1
                        scheduled_any = True
            
            # Phase 3: Advance time
            if not scheduled_any:
                # Find next event time
                next_time = current_time + 1.0
                
                # Check when resources become available
                for queue in self.resource_queues.values():
                    if queue.available_time > current_time:
                        next_time = min(next_time, queue.available_time)
                
                # Check when tasks can be scheduled again
                for task in self.tasks.values():
                    next_schedule_time = task.last_execution_time + task.min_interval_ms
                    if next_schedule_time > current_time:
                        next_time = min(next_time, next_schedule_time)
                
                current_time = min(next_time, time_window)
            else:
                # Small time advance to check for new opportunities
                current_time += 0.1
        
        return self.schedule_history
    
    def get_resource_utilization(self, time_window: float) -> Dict[str, float]:
        """Calculate resource utilization"""
        utilization = {}
        
        for resource_type, resources in self.resources.items():
            for resource in resources:
                busy_time = 0.0
                for schedule in self.schedule_history:
                    if resource.unit_id in schedule.assigned_resources.values():
                        # Simplified: assume resource is busy for entire task duration
                        busy_time += (schedule.end_time - schedule.start_time)
                
                utilization[resource.unit_id] = min(busy_time / time_window * 100, 100)
        
        return utilization
    
    def print_schedule_summary(self):
        """Print scheduling summary"""
        print("=== Scheduling Summary ===")
        print(f"NPU Resources: {len(self.resources[ResourceType.NPU])}")
        print(f"DSP Resources: {len(self.resources[ResourceType.DSP])}")
        print(f"Total Tasks: {len(self.tasks)}")
        print(f"Total Scheduled Events: {len(self.schedule_history)}")
        
        # Count by priority
        priority_counts = defaultdict(int)
        for task in self.tasks.values():
            priority_counts[task.priority.name] += 1
        
        print("\nTasks by Priority:")
        for priority in TaskPriority:
            count = priority_counts[priority.name]
            print(f"  {priority.name}: {count} tasks")
        
        # Count task scheduling
        task_schedule_count = defaultdict(int)
        task_latencies = defaultdict(list)
        
        for schedule in self.schedule_history:
            task_schedule_count[schedule.task_id] += 1
            task_latencies[schedule.task_id].append(schedule.actual_latency)
        
        print("\nTask Scheduling Details:")
        # Group by priority
        for priority in TaskPriority:
            priority_tasks = [t for t in self.tasks.values() if t.priority == priority]
            if priority_tasks:
                print(f"\n  {priority.name} Priority Tasks:")
                for task in priority_tasks:
                    count = task_schedule_count[task.task_id]
                    avg_latency = sum(task_latencies[task.task_id]) / len(task_latencies[task.task_id]) if task_latencies[task.task_id] else 0
                    achieved_fps = count / (self.schedule_history[-1].end_time / 1000) if self.schedule_history else 0
                    
                    print(f"    {task.task_id} ({task.name}):")
                    print(f"      Scheduled Count: {count}")
                    print(f"      Average Latency: {avg_latency:.1f}ms (Required: {task.latency_requirement}ms)")
                    print(f"      Achieved FPS: {achieved_fps:.1f} (Required: {task.fps_requirement})")
        
        # Resource utilization
        if self.schedule_history:
            time_window = self.schedule_history[-1].end_time
            utilization = self.get_resource_utilization(time_window)
            print("\nResource Utilization:")
            for resource_id, util in utilization.items():
                print(f"  {resource_id}: {util:.1f}%")
    
    def plot_task_overview(self, selected_bw: float = 4.0):
        """Plot task overview showing resource requirements and performance needs"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Prepare data
        task_data = []
        for task_id, task in self.tasks.items():
            resource_bw_map = {ResourceType.NPU: selected_bw, ResourceType.DSP: selected_bw}
            duration = task.get_total_duration(resource_bw_map)
            
            task_type = 'DSP+NPU' if task.uses_dsp and task.uses_npu else 'NPU-only' if task.uses_npu else 'DSP-only'
            dep_str = ','.join(task.dependencies) if task.dependencies else 'None'
            
            task_data.append({
                'id': task_id,
                'name': task.name,
                'priority': task.priority,
                'fps': task.fps_requirement,
                'latency': task.latency_requirement,
                'duration': duration,
                'type': task_type,
                'deps': dep_str
            })
        
        # Sort by priority for better visualization
        task_data.sort(key=lambda x: x['priority'].value)
        
        # Extract sorted data
        task_ids = [t['id'] for t in task_data]
        task_names = [t['name'] for t in task_data]
        priorities = [t['priority'].name for t in task_data]
        fps_requirements = [t['fps'] for t in task_data]
        latency_requirements = [t['latency'] for t in task_data]
        task_durations = [t['duration'] for t in task_data]
        task_types = [t['type'] for t in task_data]
        dependencies_str = [t['deps'] for t in task_data]
        
        x = np.arange(len(task_ids))
        
        # Plot 1: Priority distribution
        priority_colors = {
            'CRITICAL': 'red',
            'HIGH': 'orange',
            'NORMAL': 'yellow',
            'LOW': 'green'
        }
        colors = [priority_colors[p] for p in priorities]
        
        ax1.bar(x, [1]*len(x), color=colors)
        ax1.set_xlabel('Task', fontsize=14)
        ax1.set_ylabel('Priority Level', fontsize=14)
        ax1.set_title('Task Priority Distribution', fontsize=16, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'{tid}\n{name}' for tid, name in zip(task_ids, task_names)], 
                           rotation=45, ha='right', fontsize=12)
        ax1.set_ylim(0, 1.5)
        
        # Add priority labels
        for i, priority in enumerate(priorities):
            ax1.text(i, 0.5, priority, ha='center', va='center', fontsize=11, fontweight='bold')
        
        # Create custom legend
        legend_elements = [patches.Patch(color=color, label=priority) 
                          for priority, color in priority_colors.items()]
        ax1.legend(handles=legend_elements, loc='upper right', fontsize=11)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Performance requirements
        width = 0.35
        bars1 = ax2.bar(x - width/2, fps_requirements, width, label='FPS Requirement', color='skyblue')
        bars2 = ax2.bar(x + width/2, latency_requirements, width, label='Latency Requirement (ms)', color='lightcoral')
        
        ax2.set_xlabel('Task', fontsize=14)
        ax2.set_ylabel('Value', fontsize=14)
        ax2.set_title(f'Task Performance Requirements (BW={selected_bw})', fontsize=16, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(task_ids, rotation=45, ha='right', fontsize=12)
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Add values on bars
        for bar in bars1:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}', ha='center', va='bottom', fontsize=11)
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}', ha='center', va='bottom', fontsize=11)
        
        # Plot 3: Execution time by task type
        type_colors = {'NPU-only': 'green', 'DSP+NPU': 'orange', 'DSP-only': 'blue'}
        bar_colors = [type_colors.get(t, 'gray') for t in task_types]
        
        bars3 = ax3.bar(x, task_durations, color=bar_colors)
        
        ax3.set_xlabel('Task', fontsize=14)
        ax3.set_ylabel('Execution Time (ms)', fontsize=14)
        ax3.set_title(f'Task Execution Time and Resource Type (BW={selected_bw})', fontsize=16, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(task_ids, rotation=45, ha='right', fontsize=12)
        
        # Add legend
        legend_elements = [patches.Patch(color=color, label=task_type) 
                          for task_type, color in type_colors.items()]
        ax3.legend(handles=legend_elements, loc='upper right', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # Add values on bars
        for bar, duration in zip(bars3, task_durations):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{duration:.1f}', ha='center', va='bottom', fontsize=11)
        
        # Plot 4: Dependencies
        ax4.text(0.5, 0.95, 'Task Dependencies', ha='center', va='top', 
                fontsize=16, fontweight='bold', transform=ax4.transAxes)
        
        y_pos = 0.85
        for i, (tid, deps, priority) in enumerate(zip(task_ids, dependencies_str, priorities)):
            color = priority_colors[priority]
            ax4.text(0.1, y_pos - i*0.08, f'{tid}:', fontsize=12, fontweight='bold', 
                    transform=ax4.transAxes, color=color)
            ax4.text(0.3, y_pos - i*0.08, f'Depends on {deps}', fontsize=12, 
                    transform=ax4.transAxes)
        
        ax4.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def plot_pipeline_schedule(self, time_window: float = None, show_first_n: int = None):
        """Plot pipeline schedule Gantt chart with priority colors"""
        if not self.schedule_history:
            print("No schedule history, please run scheduling algorithm first")
            return
        
        # Determine time window
        if time_window is None:
            time_window = max(s.end_time for s in self.schedule_history) * 1.1
        
        # Prepare resource list
        all_resources = []
        resource_types = []
        for res_type in [ResourceType.NPU, ResourceType.DSP]:
            for resource in self.resources[res_type]:
                all_resources.append(resource.unit_id)
                resource_types.append(res_type.value)
        
        # Create resource index mapping
        resource_to_y = {res_id: i for i, res_id in enumerate(all_resources)}
        
        # Priority color mapping
        priority_colors = {
            TaskPriority.CRITICAL: 'red',
            TaskPriority.HIGH: 'orange',
            TaskPriority.NORMAL: 'yellow',
            TaskPriority.LOW: 'lightgreen'
        }
        
        # Create figure
        fig, ax = plt.subplots(figsize=(18, max(10, len(all_resources) * 1.2)))
        
        # Plot scheduling blocks
        schedules_to_plot = self.schedule_history[:show_first_n] if show_first_n else self.schedule_history
        
        for schedule in schedules_to_plot:
            task = self.tasks[schedule.task_id]
            color = priority_colors[task.priority]
            
            # For each task segment, plot resource usage
            for seg in task.segments:
                if seg.resource_type in schedule.assigned_resources:
                    resource_id = schedule.assigned_resources[seg.resource_type]
                    if resource_id in resource_to_y:
                        y_pos = resource_to_y[resource_id]
                        
                        # Calculate actual execution time for this segment
                        resource_unit = next((r for r in self.resources[seg.resource_type] 
                                            if r.unit_id == resource_id), None)
                        if resource_unit:
                            duration = seg.get_duration(resource_unit.bandwidth)
                            start_time = schedule.start_time + seg.start_time
                            
                            # Draw rectangle with priority color
                            rect = patches.Rectangle(
                                (start_time, y_pos - 0.4), duration, 0.8,
                                linewidth=2, edgecolor='black',
                                facecolor=color,
                                alpha=0.8
                            )
                            ax.add_patch(rect)
                            
                            # Add task label
                            if duration > 5:  # Only add label on blocks wide enough
                                ax.text(start_time + duration/2, y_pos,
                                       f'{task.task_id}', 
                                       ha='center', va='center', fontsize=11,
                                       weight='bold')
        
        # Set axes
        ax.set_ylim(-0.5, len(all_resources) - 0.5)
        ax.set_xlim(0, time_window)
        ax.set_yticks(range(len(all_resources)))
        ax.set_yticklabels([f'{res_id}\n({res_type})' for res_id, res_type 
                           in zip(all_resources, resource_types)], fontsize=12)
        ax.set_xlabel('Time (ms)', fontsize=14)
        ax.set_ylabel('Resource', fontsize=14)
        ax.set_title('Priority-Aware Task Scheduling Gantt Chart', fontsize=16, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
        ax.tick_params(axis='x', labelsize=12)
        
        # Add resource type separator lines
        current_type = resource_types[0]
        for i, res_type in enumerate(resource_types[1:], 1):
            if res_type != current_type:
                ax.axhline(y=i-0.5, color='red', linestyle='--', linewidth=2)
                current_type = res_type
        
        # Add legend with priorities
        legend_elements = []
        for priority in TaskPriority:
            legend_elements.append(
                patches.Patch(color=priority_colors[priority], 
                            label=f'{priority.name} Priority')
            )
        
        # Add task legend
        legend_elements.append(patches.Patch(color='white', label=''))  # Separator
        for task_id, task in self.tasks.items():
            legend_elements.append(
                patches.Patch(color='white', 
                            label=f'{task_id}: {task.name}')
            )
        
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=11)
        
        # Add resource utilization info
        utilization = self.get_resource_utilization(time_window)
        util_text = "Resource Utilization:\n"
        for res_id, util in utilization.items():
            util_text += f"{res_id}: {util:.1f}%\n"
        
        ax.text(1.02, 0.02, util_text, transform=ax.transAxes, 
               fontsize=11, verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.show()


# Usage example
if __name__ == "__main__":
    # Create scheduler
    scheduler = MultiResourceScheduler()
    
    # Add multiple NPU resources (different bandwidths)
    scheduler.add_npu("NPU_0", bandwidth=8.0)  # High performance NPU
    scheduler.add_npu("NPU_1", bandwidth=4.0)  # Medium performance NPU
    scheduler.add_npu("NPU_2", bandwidth=2.0)  # Low performance NPU
    
    # Add DSP resources
    scheduler.add_dsp("DSP_0", bandwidth=4.0)
    scheduler.add_dsp("DSP_1", bandwidth=4.0)
    
    # Create multiple tasks with different priorities
    # CRITICAL priority task
    task1 = NNTask("T1", "SafetyMonitor", priority=TaskPriority.CRITICAL)
    task1.set_npu_only({2.0: 20, 4.0: 12, 8.0: 8})
    task1.set_performance_requirements(fps=30, latency=30)
    scheduler.add_task(task1)
    
    # HIGH priority tasks
    task2 = NNTask("T2", "ObstacleDetection", priority=TaskPriority.HIGH)
    task2.set_dsp_npu_sequence([
        (ResourceType.DSP, {4.0: 8, 8.0: 5}, 0),
        (ResourceType.NPU, {2.0: 25, 4.0: 15, 8.0: 10}, 8),
    ])
    task2.set_performance_requirements(fps=20, latency=50)
    scheduler.add_task(task2)
    
    task3 = NNTask("T3", "LaneDetection", priority=TaskPriority.HIGH)
    task3.set_npu_only({2.0: 30, 4.0: 18, 8.0: 12})
    task3.set_performance_requirements(fps=15, latency=60)
    task3.add_dependency("T1")  # Depends on safety monitor
    scheduler.add_task(task3)
    
    # NORMAL priority tasks
    task4 = NNTask("T4", "TrafficSignRecog", priority=TaskPriority.NORMAL)
    task4.set_dsp_npu_sequence([
        (ResourceType.DSP, {4.0: 10}, 0),
        (ResourceType.NPU, {2.0: 20, 4.0: 12, 8.0: 8}, 10),
        (ResourceType.DSP, {4.0: 5}, 22),
    ])
    task4.set_performance_requirements(fps=10, latency=80)
    scheduler.add_task(task4)
    
    task5 = NNTask("T5", "PedestrianTracking", priority=TaskPriority.NORMAL)
    task5.set_npu_only({2.0: 35, 4.0: 20, 8.0: 12})
    task5.set_performance_requirements(fps=10, latency=100)
    scheduler.add_task(task5)
    
    # LOW priority tasks
    task6 = NNTask("T6", "SceneUnderstanding", priority=TaskPriority.LOW)
    task6.set_npu_only({2.0: 50, 4.0: 30, 8.0: 20})
    task6.set_performance_requirements(fps=5, latency=200)
    scheduler.add_task(task6)
    
    task7 = NNTask("T7", "MapUpdate", priority=TaskPriority.LOW)
    task7.set_dsp_npu_sequence([
        (ResourceType.DSP, {4.0: 15}, 0),
        (ResourceType.NPU, {2.0: 40, 4.0: 25, 8.0: 15}, 15),
    ])
    task7.set_performance_requirements(fps=2, latency=500)
    scheduler.add_task(task7)
    
    # More tasks to show priority effects
    for i in range(8, 12):
        priority = TaskPriority.NORMAL if i % 2 == 0 else TaskPriority.LOW
        task = NNTask(f"T{i}", f"Task_{i}", priority=priority)
        task.set_npu_only({2.0: 20+i*2, 4.0: 15+i, 8.0: 10+i//2})
        task.set_performance_requirements(fps=8, latency=150)
        scheduler.add_task(task)
    
    # Execute priority-aware scheduling
    print("Starting priority-aware scheduling...")
    schedule_results = scheduler.priority_aware_schedule(time_window=500.0)
    
    # Print results
    scheduler.print_schedule_summary()
    
    # Plot task overview with priorities
    print("\nPlotting task overview...")
    scheduler.plot_task_overview(selected_bw=4.0)
    
    # Plot scheduling Gantt chart with priority colors
    print("\nPlotting priority-aware scheduling Gantt chart...")
    scheduler.plot_pipeline_schedule(time_window=200.0)  # Show first 200ms
    
    # Print first 15 scheduling events
    print("\nFirst 15 scheduling events:")
    for i, schedule in enumerate(schedule_results[:15]):
        task = scheduler.tasks[schedule.task_id]
        print(f"{i+1}. [{task.priority.name}] {task.name} @ {schedule.start_time:.1f}-{schedule.end_time:.1f}ms, "
              f"Resources: {list(schedule.assigned_resources.values())}")