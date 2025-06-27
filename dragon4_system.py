#!/usr/bin/env python3
"""
Dragon4 Hardware System
纯硬件系统定义，不包含任何任务
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from scheduler import MultiResourceScheduler
from comprehensive_segmentation_patch import apply_comprehensive_segmentation_patch
from precision_scheduler_patch import apply_precision_scheduling_patch
from scheduler_segmentation_fix_v2 import apply_complete_segmentation_fix
from quick_fix_segmentation import apply_quick_segmentation_fix


@dataclass
class Dragon4Config:
    """Dragon4硬件配置参数"""
    npu_bandwidth: float = 120.0  # 每个NPU的带宽（GOPS）
    npu_count: int = 2
    dsp_count: int = 2           # DSP单元数量
    dsp_bandwidth: float = 40.0  # 每个DSP的带宽（GOPS）
    enable_segmentation: bool = True
    enable_precision_scheduling: bool = True
    max_segmentation_overhead_ratio: float = 0.15


class Dragon4System:
    """Dragon4双NPU硬件系统"""
    
    def __init__(self, config: Optional[Dragon4Config] = None):
        """初始化Dragon4硬件系统"""
        self.config = config or Dragon4Config()
        self.scheduler = None
        self._initialize_hardware()
        
    def _initialize_hardware(self):
        """初始化硬件资源"""
        # 创建调度器
        self.scheduler = MultiResourceScheduler(
            enable_segmentation=self.config.enable_segmentation,
            max_segmentation_overhead_ratio=self.config.max_segmentation_overhead_ratio
        )
        
        # 添加双NPU（相同带宽）
        if self.config.npu_count > 2:
            raise ValueError("Dragon4系统仅支持2个NPU")        
        
        for i in range(self.config.npu_count):
            npu_name = f"NPU_{i}"
            self.scheduler.add_npu(npu_name, bandwidth=self.config.npu_bandwidth)
        # self.scheduler.add_npu("NPU_0", bandwidth=self.config.npu_bandwidth)
        # self.scheduler.add_npu("NPU_1", bandwidth=self.config.npu_bandwidth)
        
        # 添加DSP单元
        for i in range(self.config.dsp_count):
            self.scheduler.add_dsp(f"DSP_{i}", bandwidth=self.config.dsp_bandwidth)
        
        # 应用系统补丁
        self._apply_system_patches()
        
        # 打印系统信息
        self._print_initialization_info()
        
    def _apply_system_patches(self):
        """应用系统级补丁（按照simple_seg_test.py的顺序）"""
        if self.config.enable_segmentation:
            # 1. 首先应用comprehensive patch
            print(f"✅ Applying comprehensive segmentation patch...")
            apply_comprehensive_segmentation_patch(self.scheduler)
            
            # 2. 应用V2修复
            print(f"✅ Applying segmentation fix V2...")
            apply_complete_segmentation_fix(self.scheduler)
            
            # 3. 应用quick fix（增加缓冲和成本）
            print(f"✅ Applying quick segmentation fix...")
            apply_quick_segmentation_fix(self.scheduler, buffer_ms=0.2, cost_ms=0.1)
            
        if self.config.enable_precision_scheduling:
            print(f"✅ Applying precision scheduling patch...")
            apply_precision_scheduling_patch(self.scheduler)
    
    def _print_initialization_info(self):
        """打印初始化信息"""
        print(f"\n🐉 Dragon4 Hardware System Initialized:")
        print(f"  - 2 x NPU @ {self.config.npu_bandwidth} GOPS each")
        print(f"  - {self.config.dsp_count} x DSP @ {self.config.dsp_bandwidth} GOPS each")
        print(f"  - Total NPU Bandwidth: {2 * self.config.npu_bandwidth} GOPS")
        print(f"  - Total DSP Bandwidth: {self.config.dsp_count * self.config.dsp_bandwidth} GOPS")
        print(f"  - Segmentation: {'Enabled' if self.config.enable_segmentation else 'Disabled'}")
        print(f"  - Precision Scheduling: {'Enabled' if self.config.enable_precision_scheduling else 'Disabled'}")
    
    def get_hardware_info(self) -> Dict:
        """获取硬件信息"""
        return {
            "system": "Dragon4",
            "npu": {
                "count": 2,
                "bandwidth_each": self.config.npu_bandwidth,
                "total_bandwidth": 2 * self.config.npu_bandwidth,
                "units": ["NPU_0", "NPU_1"]
            },
            "dsp": {
                "count": self.config.dsp_count,
                "bandwidth_each": self.config.dsp_bandwidth,
                "total_bandwidth": self.config.dsp_count * self.config.dsp_bandwidth,
                "units": [f"DSP_{i}" for i in range(self.config.dsp_count)]
            },
            "features": {
                "segmentation": self.config.enable_segmentation,
                "precision_scheduling": self.config.enable_precision_scheduling,
                "max_segmentation_overhead": self.config.max_segmentation_overhead_ratio
            }
        }
    
    def get_resource_names(self) -> Dict[str, List[str]]:
        """获取资源名称列表"""
        return {
            "NPU": ["NPU_0", "NPU_1"],
            "DSP": [f"DSP_{i}" for i in range(self.config.dsp_count)]
        }
    
    def schedule(self, time_window: float = 1000.0) -> List:
        """执行调度"""
        if not self.scheduler:
            raise RuntimeError("Scheduler not initialized")
        return self.scheduler.priority_aware_schedule_with_segmentation(time_window)
    
    def get_resource_utilization(self, time_window: float) -> Dict[str, float]:
        """获取资源利用率"""
        if not self.scheduler:
            return {}
        return self.scheduler.get_resource_utilization(time_window)
    
    def reset(self):
        """重置系统状态"""
        # 清空任务但保留硬件配置
        if self.scheduler:
            self.scheduler.tasks.clear()
            self.scheduler.schedule_history.clear()
            self.scheduler.active_bindings.clear()
            
            # 重置资源队列
            for queue in self.scheduler.resource_queues.values():
                queue.available_time = 0.0
                if hasattr(queue, 'release_binding'):
                    queue.release_binding()
    
    def print_hardware_summary(self):
        """打印硬件摘要"""
        print("\n=== Dragon4 Hardware Summary ===")
        info = self.get_hardware_info()
        
        print(f"NPU Subsystem:")
        print(f"  - Count: {info['npu']['count']}")
        print(f"  - Bandwidth: {info['npu']['bandwidth_each']} GOPS each")
        print(f"  - Total: {info['npu']['total_bandwidth']} GOPS")
        
        print(f"\nDSP Subsystem:")
        print(f"  - Count: {info['dsp']['count']}")
        print(f"  - Bandwidth: {info['dsp']['bandwidth_each']} GOPS each")
        print(f"  - Total: {info['dsp']['total_bandwidth']} GOPS")
        
        print(f"\nSystem Features:")
        for feature, enabled in info['features'].items():
            status = "Enabled" if enabled else "Disabled" if isinstance(enabled, bool) else str(enabled)
            print(f"  - {feature}: {status}")


if __name__ == "__main__":
    # 硬件系统测试
    print("=== Dragon4 Hardware System Test ===\n")
    
    # 测试默认配置
    print("1. Default Configuration:")
    system1 = Dragon4System()
    system1.print_hardware_summary()
    
    # 测试高性能配置
    print("\n\n2. High Performance Configuration:")
    high_perf_config = Dragon4Config(
        npu_bandwidth=240.0,
        dsp_count=4,
        dsp_bandwidth=80.0
    )
    system2 = Dragon4System(high_perf_config)
    system2.print_hardware_summary()
    
    # 测试低功耗配置
    print("\n\n3. Low Power Configuration:")
    low_power_config = Dragon4Config(
        npu_bandwidth=60.0,
        dsp_count=1,
        dsp_bandwidth=20.0
    )
    system3 = Dragon4System(low_power_config)
    system3.print_hardware_summary()
