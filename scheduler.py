from typing import List, Dict
from collections import defaultdict
from enums import ResourceType, TaskPriority
from models import ResourceUnit, TaskScheduleInfo
from task import NNTask
from priority_queue import ResourcePriorityQueues

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
