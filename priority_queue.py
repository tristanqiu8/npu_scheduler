from typing import Optional, Tuple
from collections import deque
from enums import TaskPriority
from task import NNTask

class ResourcePriorityQueues:
    """Priority queues for a single resource with binding support"""
    
    def __init__(self, resource_id: str):
        self.resource_id = resource_id
        # Create a queue for each priority level
        self.queues = {
            priority: deque() for priority in TaskPriority
        }
        self.available_time = 0.0  # When resource becomes available
        self.bound_until = 0.0     # When resource binding expires
        self.binding_task_id = None  # Task ID that has bound this resource
    
    def add_task(self, task: NNTask, ready_time: float):
        """Add task to appropriate priority queue"""
        self.queues[task.priority].append((task, ready_time))
    
    def is_bound_to_other_task(self, task_id: str, current_time: float) -> bool:
        """Check if resource is bound to a different task"""
        return (self.bound_until > current_time and 
                self.binding_task_id is not None and 
                self.binding_task_id != task_id)
    
    def bind_resource(self, task_id: str, end_time: float):
        """Bind resource to a specific task until end_time"""
        self.binding_task_id = task_id
        self.bound_until = end_time
    
    def release_binding(self):
        """Release resource binding"""
        self.binding_task_id = None
        self.bound_until = 0.0
    
    def get_next_task(self, current_time: float) -> Optional[Tuple[NNTask, float]]:
        """Get next task to execute based on priority"""
        # Check queues from highest to lowest priority
        for priority in TaskPriority:
            queue = self.queues[priority]
            # Remove tasks that are not ready yet
            ready_tasks = []
            while queue:
                task, ready_time = queue.popleft()
                if (ready_time <= current_time and 
                    task.last_execution_time + task.min_interval_ms <= current_time and
                    not self.is_bound_to_other_task(task.task_id, current_time)):
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
    
    def has_higher_priority_tasks(self, priority: TaskPriority, current_time: float, task_id: str = None) -> bool:
        """Check if there are higher priority tasks waiting"""
        for p in TaskPriority:
            if p < priority:  # Higher priority
                # Check if any task in this queue is ready
                for task, ready_time in self.queues[p]:
                    if (ready_time <= current_time and 
                        task.last_execution_time + task.min_interval_ms <= current_time and
                        not self.is_bound_to_other_task(task.task_id, current_time)):
                        return True
        return False