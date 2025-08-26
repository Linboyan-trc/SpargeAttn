import os
from pathlib import Path
import subprocess
from packaging.version import parse, Version
from typing import List, Set
import warnings
from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

####################################################################################################
# 1. 支持的GPU有Ampere:8.0 A100, 8.6 RTX 3090, RTX 3080, 8.7
# 1. 支持的GPU有Ada:8.9 RTX 4090, RTX 4080
# 1. 支持的GPU有Hopper:9.0 H100
SUPPORTED_ARCHS = {"8.0", "8.6", "8.7", "8.9", "9.0"}

HAS_SM90 = False

####################################################################################################
# 1. CXX_FLAGS, NVCC_FLAGS
# 1.1 设置基础编译参数
CXX_FLAGS = [
    "-std=c++17", "-g", "-O3", 
    "-fopenmp", "-lgomp", 
    "-DENABLE_BF16"
]
NVCC_FLAGS = [
    "-std=c++17", "-O3",
    "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
    "--use_fast_math", "--threads=8", "-Xptxas=-v", "-diag-suppress=174",
]
ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
CXX_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]
NVCC_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]

####################################################################################################
# 1. GPU架构, CUDA版本, 最终作为NVCC_FLAGS的编译参数
# 1.1 查看CUDA_HOME位置
# 1.1 就是CUDA Toolkit的位置，在安装torch库的时候会自动找本地CUDA Toolkit的位置并记录下来
if CUDA_HOME is None:
    raise RuntimeError(
        "Cannot find CUDA_HOME. CUDA must be available to build the package.")

# 1.2 获取需要支持的GPU SM_XX架构
# 1.2.1 需要在运行前手动设置环境变量，也就是`TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9+PTX" python setup.py install`
def get_torch_arch_list() -> Set[str]:
    # 1.2.1 获取手动设置的需要支持的架构的环境变量
    env_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
    if env_arch_list is None:
        return set()

    # 1.2.2 然后转列表
    torch_arch_list = set(env_arch_list.replace(" ", ";").split(";"))
    if not torch_arch_list:
        return set()

    # 1.2.3 本项目支持的架构有8.0～9.0，8.0+PTX～9.0+PTX
    valid_archs = SUPPORTED_ARCHS.union({s + "+PTX" for s in SUPPORTED_ARCHS})

    # 1.2.4 对于用于指定的GPU架构，和本项目支持的GPU架构，取交集
    arch_list = torch_arch_list.intersection(valid_archs)
    
    # 1.2.5 如果是空集，说明本项目不支持用户指定的GPU架构
    if not arch_list:
        raise RuntimeError(
            "None of the CUDA architectures in `TORCH_CUDA_ARCH_LIST` env "
            f"variable ({env_arch_list}) is supported. "
            f"Supported CUDA architectures are: {valid_archs}.")
    
    # 1.2.6 如果不是空集，先告诉用户指定的架构中哪些不支持，哪些支持
    invalid_arch_list = torch_arch_list - valid_archs
    if invalid_arch_list:
        warnings.warn(
            f"Unsupported CUDA architectures ({invalid_arch_list}) are "
            "excluded from the `TORCH_CUDA_ARCH_LIST` env variable "
            f"({env_arch_list}). Supported CUDA architectures are: "
            f"{valid_archs}.")
        
    # 1.2.7 最后返回用户指定且本项目支持的架构
    return arch_list

# 1.3 获取CUDA Toolkit版本
# 1.3.1 没有命令直接获取CUDA Toolkit版本，但是有命令直接获取NVCC版本，NVCC的版本就是CUDA Toolkit的版本
def get_nvcc_cuda_version(cuda_dir: str) -> Version:
    # 1.3.1 执行"/usr/cuda/bin/nvcc -v"获取NVCC版本
    # 1.3.1 找到返回的输出中"release"的位置，后面就是版本号
    nvcc_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"],universal_newlines=True)
    output = nvcc_output.split()
    release_idx = output.index("release") + 1
    nvcc_cuda_version = parse(output[release_idx].split(",")[0])
    return nvcc_cuda_version

# 1.4 获取用户指定且本项目支持的架构
compute_capabilities = get_torch_arch_list()

# 1.5 如果用户没有通过环境变量TORCH_CUDA_ARCH_LIST指定架构，那么检测当前本地连接的GPU
if not compute_capabilities:
    # 1.5.1 获取GPU数量
    device_count = torch.cuda.device_count()

    # 1.5.2 获取每个GPU的主版本号和次版本号，比如8.6中8是主版本号，6是次版本号
    for i in range(device_count):
        major, minor = torch.cuda.get_device_capability(i)
        if major < 8:
            raise RuntimeError("GPUs with compute capability below 8.0 are not supported.")
        compute_capabilities.add(f"{major}.{minor}")

# 1.6 如果还是没有，说明本地没有连接GPU，本项目必须在GPU上运行
if not compute_capabilities:
    raise RuntimeError("No GPUs found. Please specify the target GPU architectures or build on a machine with GPUs.")

# 1.7 获取本地CUDA Toolkit的版本
nvcc_cuda_version = get_nvcc_cuda_version(CUDA_HOME)

# 1.8 本项目从Ampere系列起开始适配，CUDA Toolkit版本必须>=12.0
if nvcc_cuda_version < Version("12.0"):
    raise RuntimeError("CUDA 12.0 or higher is required to build the package.")

# 1.9 如果需要适配Ada，Hopper，那么CUDA Toolkit版本必须>=12.4
if nvcc_cuda_version < Version("12.4"):
    if any(cc.startswith("8.9") for cc in compute_capabilities):
        raise RuntimeError("CUDA 12.4 or higher is required for compute capability 8.9.")
    if any(cc.startswith("9.0") for cc in compute_capabilities):
        raise RuntimeError("CUDA 12.4 or higher is required for compute capability 9.0.")

# 2.0 增加NVCC_FLAGS编译参数
for capability in compute_capabilities:
    # 2.1 拼接得到'80', '89', '90'
    num = capability[0] + capability[2]

    # 2.2 对于'90'架构需要特殊处理
    # 2.2.1 包括num改成'90a'
    # 2.2.2 HAS_SM90设置为True, 以增加sm_90相关的CUDA代码
    # 2.2.3 CXX_FLAGS增加宏HAS_SM90, 来对源码中被#ifdef SM_90的部分编译
    if num == '90':
        num = '90a'
        HAS_SM90 = True
        CXX_FLAGS += ["-DHAS_SM90"]

    # 2.3 NVCC_FLAGS参数`-gencode`, `arch=compute_90a,code=sm_90a`
    NVCC_FLAGS += ["-gencode", f"arch=compute_{num},code=sm_{num}"]

    # 2.4 如果还需要生成PTX中间代码
    # 2.4.1 NVCC_FLAGS参数`-gencode`, `arch=compute_90a,code=compute_90a`
    if capability.endswith("+PTX"):
        NVCC_FLAGS += ["-gencode", f"arch=compute_{num},code=compute_{num}"]

####################################################################################################
# 1. 遍历传入目录下所有.py文件，并运行
def run_instantiations(src_dir: str):
    base_path = Path(src_dir)
    py_files = [
        path for path in base_path.rglob('*.py')
        if path.is_file()
    ]

    for py_file in py_files:
        print(f"Running: {py_file}")
        os.system(f"python {py_file}")

# 2. 遍历传入目录下所有.cu 文件，将其路径组织成列表返回
def get_instantiations(src_dir: str):
    base_path = Path(src_dir)
    return [
        os.path.join(src_dir, str(path.relative_to(base_path)))
        for path in base_path.rglob('*')
        if path.is_file() and path.suffix == ".cu"
    ]

run_instantiations("csrc/qattn/instantiations_sm80")
run_instantiations("csrc/qattn/instantiations_sm89")
run_instantiations("csrc/qattn/instantiations_sm90")

sources = [ "csrc/qattn/pybind.cpp",
            "csrc/qattn/qk_int_sv_f16_cuda_sm80.cu",
            "csrc/qattn/qk_int_sv_f8_cuda_sm89.cu",] 
sources += get_instantiations("csrc/qattn/instantiations_sm80")
sources += get_instantiations("csrc/qattn/instantiations_sm89")
if HAS_SM90:
    sources += ["csrc/qattn/qk_int_sv_f8_cuda_sm90.cu", ]
    sources += get_instantiations("csrc/qattn/instantiations_sm90")

####################################################################################################
# 1. 需要提前编译
# 1.1 作为模块"spas_sage_attn._qattn"被其他模块引用，此模块挂靠在本项目下，所以命名是<项目名>.<cuda模块名>
ext_modules = []
qattn_extension = CUDAExtension(
    # 1.1 模块名 + 源代码 + 编译参数
    name="spas_sage_attn._qattn",
    sources=sources,
    extra_compile_args={
        "cxx": CXX_FLAGS,
        "nvcc": NVCC_FLAGS,
    },

    # 1.2 用于链接CUDA Driver，编译的时候要用到其中的库，一般情况下的CUDA代码用不到这个库
    extra_link_args=['-lcuda'],
)
ext_modules.append(qattn_extension)

# 2. 需要提前编译
# 2.1 作为模块"spas_sage_attn._fused"被其他模块引用，此模块挂靠在本项目下，所以命名是<项目名>.<cuda模块名>
fused_extension = CUDAExtension(
    # 2.1 模块名 + 源代码 + 编译参数
    name="spas_sage_attn._fused",
    sources=["csrc/fused/pybind.cpp", "csrc/fused/fused.cu"],
    extra_compile_args={
        "cxx": CXX_FLAGS,
        "nvcc": NVCC_FLAGS,
    },
)
ext_modules.append(fused_extension)

####################################################################################################
# 1. 设置
setup(
    # 1. 把整个项目打包成"spas_sage_attn"模块
    # 1. 自动查找并包含SpargeAttn项目中的Python包，也就是有__init__.py的目录
    name='spas_sage_attn', 
    version='0.1.0',  
    author='Jintao Zhang, Chendong Xiang, Haofeng Huang',  
    author_email='jt-zhang6@gmail.com', 
    description='Accurate and efficient Sparse SageAttention.',  
    long_description=open('README.md', encoding='utf-8').read(),  
    long_description_content_type='text/markdown', 
    url='https://github.com/thu-ml/SpargeAttn', 
    license='BSD 3-Clause License', 
    python_requires='>=3.9', 
    classifiers=[  
        'Development Status :: 3 - Alpha', 
        'Intended Audience :: Developers',  
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3', 
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
    ],
    packages=find_packages(),  

    # 2. 指定整个项目运行所需要提前编译的包
    # 2. ext_modules指定运行项目前需要提前编译的模块
    # 2. 如果不执行setup.py编译一下，直接运行本项目，会无法运行
    # 2.1 ext_modules起初为c/c++代码设计，用意是setup会自动用gcc/g++编译好ext_modules指定的c/c++代码，然后让python去调用编译好的代码，获得高性能
    # 2.1 本质就是对于需要高性能计算的部分，用c/c++的编译方式代替python代码的解释方式，获得更高性能的汇编代码
    ext_modules=ext_modules,

    # 3. 指定编译器，来编译ext_modules中指定的需要提前编译的模块
    # 3.1 如果是纯c/c++代码，只需要给出ext_modules就可以
    # 3.2 但是这里会涉及cuda代码，所以需要专门指定一下编译器，使得可以同时编译c/c++和cuda代码
    cmdclass={"build_ext": BuildExtension},
)
