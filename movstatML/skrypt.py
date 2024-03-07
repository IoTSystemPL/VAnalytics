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


def run_command(cmd: str):
    print(cmd)
    return os.system(cmd)


def get_host_compiler() -> str:
    if get_platform() == 'linux':
        return 'g++'
    if get_platform() == 'windows':
        return 'cl'


def build_cuda_lib_msvc(build_target: str) -> str:
    common_includes = ' -default-stream per-thread -O3 -DUSE_CUDA=1'# -DDLL_API_EXPORTS'
    compiler = 'nvcc -ccbin cl -m64'

    def get_sm_archs(min_arch: int) -> str:
        sm_archs = [10, 11, 12, 13, 20, 21, 30, 32, 35, 37, 50, 52, 53, 60, 61, 62, 70, 72, 75]
        result = ''
        for a in sm_archs:
            if a >= min_arch:
                result += ' -gencode=arch=compute_' + str(a) + ',code=sm_' + str(a)
        return result

    compiler += ' -arch=sm_61' #+ get_sm_archs(61)
    if build_target == 'debug':
        compiler += ' -Xcompiler=\"/MDd\"'
    else:
        compiler += ' -Xcompiler=\"/MD\" -DNDEBUG'

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

    cmd = compiler + ' -lib -o ' + get_lib_name() + '.lib'
    for sourcefile in lista_cu:
        file_name = sourcefile[1][:-3]
        cmd += ' objects/' + file_name + '.o' + ' objects/' + file_name + '.dlink.o'
    commands += cmd + '\n'
    commands += 'cd ../movstatML/' + '\n'
    return commands


def build_cpu_lib_msvc(build_target: str) -> str:
    compiler = 'cl'
    common_options = ' /openmp /O2 /EHsc -DUSE_CUDA=0'# -DDLL_API_EXPORTS'
    if build_target == 'debug':
        common_options += ' /MDd /Zi'
    else:
        common_options += ' /MD /O2 -DNDEBUG'

    commands = 'cd ../cpuml/' + '\n'
    commands += 'mkdir objects' + '\n'
    commands += 'call \"C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Auxiliary\\Build\\vcvars64.bat\"' + '\n'
    # separately compile different SIMD variants
    lista_cpp = find_sources('../cpuml/', '.cpp', '.h', ['stdafx.cpp'])
    object_list = []
    simd_variants = {'def_': '', 'sse_': 'sse2', 'avx_': 'AVX', 'avx2_': 'AVX2'}
    for simd in simd_variants:
        for sourcefile in lista_cpp:
            file_name = sourcefile[1][:-4]
            if file_name.startswith(simd):
                cmd = compiler + common_options + ' /c /Foobjects/' + file_name + '.o' + ' ' + sourcefile[1]
                if simd != 'def_' and simd != 'sse_':
                    cmd += ' /arch:' + simd_variants[simd]
                commands += cmd + '\n'
                object_list.append('objects/' + file_name + '.o')

    def get_lib_name() -> str:
        if build_target == 'debug':
            return 'cpu_math_d'
        else:
            return 'cpu_math'

    cmd = 'lib /OUT:' + get_lib_name() + '.lib'
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
    cmd = build_cpu_lib_msvc(target)
    if not cpu_only:
        cmd += build_cuda_lib_msvc(target)

    f = open('build_lib_' + target + '.bat', 'w')
    f.write(cmd)
    f.close()

    run_command('build_lib_' + target + '.bat')
