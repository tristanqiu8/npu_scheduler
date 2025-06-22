#!/usr/bin/env python3
"""
NPU Scheduler Package Setup
"""

from setuptools import setup, find_packages
import os

# 读取README文件
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "NPU Scheduler - Multi-resource Neural Network Task Scheduler"

# 读取版本信息
def get_version():
    version_file = os.path.join(os.path.dirname(__file__), 'core', '__init__.py')
    if os.path.exists(version_file):
        with open(version_file, 'r') as f:
            for line in f:
                if line.startswith('__version__'):
                    return line.split('=')[1].strip().strip('"\'')
    return "2.0.0"

# 基础依赖
REQUIRED_PACKAGES = [
    'numpy>=1.21.0',
    'matplotlib>=3.5.0',
]

# 可选依赖组
EXTRAS_REQUIRE = {
    'optimization': [
        'scipy>=1.7.0',
        'pandas>=1.3.0',
    ],
    'dev': [
        'pytest>=6.2.0',
        'pytest-cov>=2.12.0',
        'black>=21.0.0',
        'flake8>=3.9.0',
    ],
    'docs': [
        'sphinx>=4.0.0',
        'sphinx-rtd-theme>=0.5.0',
    ],
    'profiling': [
        'memory-profiler>=0.58.0',
        'psutil>=5.8.0',
    ]
}

# 添加 'all' 选项，包含所有可选依赖
EXTRAS_REQUIRE['all'] = []
for deps in EXTRAS_REQUIRE.values():
    EXTRAS_REQUIRE['all'].extend(deps)

setup(
    name="npu-scheduler",
    version=get_version(),
    
    # 基本信息
    description="Multi-resource Neural Network Task Scheduler for NPU/DSP systems",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    
    # 作者信息
    author="NPU Scheduler Team",
    author_email="npu-scheduler@example.com",
    
    # 项目链接
    url="https://github.com/your-username/npu-scheduler",
    project_urls={
        "Bug Tracker": "https://github.com/your-username/npu-scheduler/issues",
        "Documentation": "https://npu-scheduler.readthedocs.io/",
        "Source Code": "https://github.com/your-username/npu-scheduler",
    },
    
    # 包信息
    packages=find_packages(),
    include_package_data=True,
    
    # Python版本要求
    python_requires=">=3.8",
    
    # 依赖
    install_requires=REQUIRED_PACKAGES,
    extras_require=EXTRAS_REQUIRE,
    
    # 分类信息
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Hardware",
    ],
    
    # 关键词
    keywords=[
        "npu", "scheduler", "neural-network", "dsp", 
        "resource-allocation", "task-scheduling", "optimization"
    ],
    
    # 命令行入口点
    entry_points={
        'console_scripts': [
            'npu-scheduler=main:main',
        ],
    },
    
    # 包数据
    package_data={
        'examples': ['*.md', '*.json'],
        'tests': ['*.json', '*.csv'],
    },
    
    # 数据文件
    data_files=[
        ('configs', ['requirements.txt']),
    ],
    
    # 开发状态
    zip_safe=False,
)
