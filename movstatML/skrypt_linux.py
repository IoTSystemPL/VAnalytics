import os
import sys


def get_platform() -> str:
    platforms = {'linux1': 'Linux',
                 'linux2': 'Linux',
                 'darwin': 'OS X',
                 'win32': 'Windows'}
    if sys.platform not in platforms:
        return sys.platform
    else:
        return platforms[sys.platform]


def find_sources(path: str, source_extension: str, header_extension: str, exclude: list = []) -> list:
    try:
        lista = os.listdir(path)
    except Exception as e:
        return []
    result = []
    for f_name in lista:
        if f_name.endswith(source_extension) and f_name not in exclude:
            result.append([path, f_name])
        else:
            if not f_name.endswith(header_extension) and f_name not in exclude:
                result += find_sources(path + f_name + '/', source_extension, header_extension)
    return result


def build_cuda_lib_gcc(build_target: str) -> str:
    common_includes = ' -default-stream per-thread -DUSE_CUDA=1 -I/usr/local/cuda/include -I/usr/local/cuda-10.2/include'
    compiler = 'nvcc -ccbin g++ -m64 -std=c++11'

    def get_sm_archs(min_arch: int) -> str:
        sm_archs = [10, 11, 12, 13, 20, 21, 30, 32, 35, 37, 50, 52, 53, 60, 61, 62, 70, 72, 75]
        result = ''
        for a in sm_archs:
            if a >= min_arch:
                result += ' -gencode=arch=compute_' + str(a) + ',code=sm_' + str(a)
        return result

    compiler += ' -arch=sm_61'  # + get_sm_archs(61)
    if build_target == 'debug':
        compiler += ' -O0 --device-debug'
    else:
        compiler += ' -O3 -DNDEBUG'

    commands = 'cd ../cudaml/' + '\n'
    commands += 'mkdir objects' + '\n'
    lista_cu = find_sources('../cudaml/', '.cu', '.cuh')
    for sourcefile in lista_cu:
        file_name = sourcefile[1][:-3]
        commands += compiler + common_includes + ' -dc -o objects/' + file_name + '.o' + ' ' + sourcefile[1] + '\n'
        commands += compiler + common_includes + ' -dlink -o objects/' + file_name + '.dlink.o' + ' objects/' + file_name + '.o' + '\n'

    def get_lib_name() -> str:
        if build_target == 'debug':
            return 'cuda_math_d'
        else:
            return 'cuda_math'

    cmd = compiler + ' -lib -o ' + get_lib_name() + '.a'
    for sourcefile in lista_cu:
        file_name = sourcefile[1][:-3]
        cmd += ' objects/' + file_name + '.o' + ' objects/' + file_name + '.dlink.o'
    commands += cmd + '\n'
    commands += 'cd ../movstatML/' + '\n'
    return commands


def build_cpu_lib_gcc(build_target: str) -> str:
    compiler = 'g++ -std=c++17 -fopenmp -DUSE_CUDA=0'
    if build_target == 'debug':
        compiler += ' -O0 -g3'
    else:
        compiler += ' -O3 -DNDEBUG'

    commands = 'cd ../cpuml/' + '\n'
    commands += 'mkdir objects' + '\n'

    lista_cpp = find_sources('../cpuml/', '.cpp', '.h', ['stdafx.cpp'])
    object_list = []

    def get_simd_flag(filename: str) -> str:
        simd_variants = {'def_': '',
                         'sse_': '-msse',
                         'sse2_': '-msse2',
                         'sse3_': '-msse3',
                         'ssse3_': '-mssse3',
                         'sse41_': '-msse41',
                         'avx_': '-mavx',
                         'avx2_': '-mavx2'}
        for key in simd_variants:
            if filename.startswith(key):
                return simd_variants[key]
        return ''

    for sourcefile in lista_cpp:
        file_name = sourcefile[1][:-4]
        cmd = compiler + ' -c -o objects/' + file_name + '.o' + ' ' + sourcefile[1]
        cmd += ' ' + get_simd_flag(file_name)
        commands += cmd + '\n'
        object_list.append('objects/' + file_name + '.o')

    def get_lib_name() -> str:
        if build_target == 'debug':
            return 'cpu_math_d'
        else:
            return 'cpu_math'

    cmd = 'ar rcs -o ' + get_lib_name() + '.a'
    for i in range(len(object_list)):
        cmd += ' ' + object_list[i]
    commands += cmd + '\n'
    commands += 'cd ../movstatML/' + '\n'
    return commands


def build_lib_gcc(build_target: str, cpu_only: bool) -> str:
    compiler = 'g++ -std=c++17 -fopenmp'
    if cpu_only:
        compiler += ' -DUSE_CUDA=0'
    else:
        compiler += ' -DUSE_CUDA=1 -I/usr/local/cuda/include -I/usr/local/cuda-10.2/include'
    if build_target == 'debug':
        compiler += ' -O0 -g3'
    else:
        compiler += ' -O3 -DNDEBUG'

    compiler += ' -I x86_64-linux-gnu -I x86_64-linux-gnu/openblas-pthread -I /usr/local/include/opencv4'

    commands = 'cd ../libml/' + '\n'
    commands += 'mkdir objects' + '\n'

    lista_cpp = find_sources('../libml/', '.cpp', '.h', ['stdafx.cpp'])
    object_list = []
    for sourcefile in lista_cpp:
        file_name = sourcefile[1][:-4]
        cmd = compiler + ' -c -o objects/' + file_name + '.o' + ' ' + sourcefile[1]
        commands += cmd + '\n'
        object_list.append('objects/' + file_name + '.o')

    def get_lib_name() -> str:
        tmp = 'ml_runtime'
        if cpu_only:
            tmp += '_cpu'
        if build_target == 'debug':
            tmp += '_d'
        return tmp

    cmd = 'ar rcs -o ../' + get_lib_name() + '.a'
    for i in range(len(object_list)):
        cmd += ' ' + object_list[i]
    commands += cmd + '\n'
    commands += 'cd ../movstatML/' + '\n'
    return commands


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('expected target argument')
        exit(-1)

    cpu_only = False
    if len(sys.argv) == 3:
        cpu_only = (sys.argv[2] == 'cpu_only')

    target = sys.argv[1]
    cmd = build_cpu_lib_gcc(target)
    if not cpu_only:
        cmd += build_cuda_lib_gcc(target)

    cmd += build_lib_gcc(target, cpu_only)
    print(cmd)
    os.system(cmd)

    print('build finished')

