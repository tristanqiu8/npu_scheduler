#!/usr/bin/env python3
"""
全局配置管理模块
提供统一的配置管理和预设配置
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import argparse


@dataclass
class SchedulerConfig:
    """调度器配置类"""
    
    # 核心功能开关
    enable_segmentation: bool = False
    enable_validation: bool = True
    apply_patches: bool = True
    
    # 可视化配置
    visualization_style: str = 'elegant'
    export_chrome_tracing: bool = False
    
    # 性能配置
    default_time_window: float = 500.0
    max_optimization_iterations: int = 10
    
    # 调试配置
    verbose_logging: bool = False
    show_detailed_metrics: bool = True
    
    @classmethod
    def for_production(cls) -> 'SchedulerConfig':
        """生产环境配置 - 稳定性优先"""
        return cls(
            enable_segmentation=False,  # 暂时禁用分段功能
            enable_validation=True,
            apply_patches=True,
            visualization_style='elegant',
            export_chrome_tracing=True,
            verbose_logging=False,
            show_detailed_metrics=True
        )
    
    @classmethod
    def for_development(cls) -> 'SchedulerConfig':
        """开发环境配置 - 功能完整"""
        return cls(
            enable_segmentation=True,
            enable_validation=True,
            apply_patches=True,
            visualization_style='elegant',
            export_chrome_tracing=True,
            verbose_logging=True,
            show_detailed_metrics=True
        )
    
    @classmethod
    def for_testing(cls) -> 'SchedulerConfig':
        """测试环境配置 - 快速执行"""
        return cls(
            enable_segmentation=False,
            enable_validation=True,
            apply_patches=True,
            visualization_style='elegant',
            export_chrome_tracing=False,
            verbose_logging=False,
            show_detailed_metrics=False,
            default_time_window=100.0,
            max_optimization_iterations=3
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SchedulerConfig':
        """从字典创建配置"""
        return cls(**config_dict)


def parse_command_line_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='NPU Scheduler - 多资源神经网络任务调度器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python main.py --mode basic                    # 基础演示
  python main.py --mode optimization             # 优化演示
  python main.py --mode tutorial                 # 教程演示
  python main.py --mode basic --segmentation     # 启用分段功能
  python main.py --config production             # 使用生产配置
        """
    )
    
    # 运行模式 - 更新为支持tutorial
    parser.add_argument(
        '--mode', 
        choices=['basic', 'optimization', 'tutorial'], 
        default='basic',
        help='运行模式选择'
    )
    
    # 预设配置
    parser.add_argument(
        '--config',
        choices=['production', 'development', 'testing'],
        default='development',
        help='预设配置选择'
    )
    
    # 功能开关
    parser.add_argument(
        '--segmentation', 
        action='store_true',
        help='启用网络分段功能'
    )
    
    parser.add_argument(
        '--no-patches', 
        action='store_true',
        help='禁用调度器补丁'
    )
    
    parser.add_argument(
        '--no-validation', 
        action='store_true',
        help='禁用调度验证'
    )
    
    # 输出控制
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='详细输出模式'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='静默输出模式'
    )
    
    parser.add_argument(
        '--export-trace',
        action='store_true',
        help='导出Chrome追踪文件'
    )
    
    # 性能参数
    parser.add_argument(
        '--time-window',
        type=float,
        default=500.0,
        help='调度时间窗口（毫秒）'
    )
    
    parser.add_argument(
        '--iterations',
        type=int,
        default=10,
        help='优化迭代次数'
    )
    
    return parser.parse_args()


def create_config_from_args(args: argparse.Namespace) -> SchedulerConfig:
    """从命令行参数创建配置"""
    
    # 获取预设配置
    if args.config == 'production':
        config = SchedulerConfig.for_production()
    elif args.config == 'development':
        config = SchedulerConfig.for_development()
    elif args.config == 'testing':
        config = SchedulerConfig.for_testing()
    else:
        config = SchedulerConfig()
    
    # 应用命令行覆盖
    if args.segmentation:
        config.enable_segmentation = True
    
    if args.no_patches:
        config.apply_patches = False
    
    if args.no_validation:
        config.enable_validation = False
    
    if args.verbose:
        config.verbose_logging = True
        config.show_detailed_metrics = True
    
    if args.quiet:
        config.verbose_logging = False
        config.show_detailed_metrics = False
    
    if args.export_trace:
        config.export_chrome_tracing = True
    
    # 性能参数
    config.default_time_window = args.time_window
    config.max_optimization_iterations = args.iterations
    
    return config


# 全局配置实例（可选）
_global_config: Optional[SchedulerConfig] = None


def get_global_config() -> SchedulerConfig:
    """获取全局配置实例"""
    global _global_config
    if _global_config is None:
        _global_config = SchedulerConfig.for_development()
    return _global_config


def set_global_config(config: SchedulerConfig) -> None:
    """设置全局配置实例"""
    global _global_config
    _global_config = config


if __name__ == "__main__":
    # 测试配置功能
    print("=== 配置测试 ===")
    
    # 预设配置测试
    for config_name, config_func in [
        ("生产环境", SchedulerConfig.for_production),
        ("开发环境", SchedulerConfig.for_development),
        ("测试环境", SchedulerConfig.for_testing)
    ]:
        config = config_func()
        print(f"\n{config_name}配置:")
        for key, value in config.to_dict().items():
            print(f"  {key}: {value}")
    
    # 命令行参数测试
    print("\n=== 命令行参数示例 ===")
    print("python main.py --mode basic --segmentation --verbose")
    print("python main.py --mode tutorial --quiet")
    print("python main.py --config production --export-trace")
