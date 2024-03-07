import requests
import tarfile
import shutil
import os


def mkdir(path: str) -> None:
    if not os.path.exists(path):
        os.mkdir(path)


def download_openblas() -> None:
    source = 'http://repo.msys2.org/mingw/x86_64/'
    name = 'mingw-w64-x86_64-openblas-0.3.9-1-any.pkg.tar.xz'
    r = requests.get(source + name)
    with open(name, 'wb') as f:
        f.write(r.content)
    with tarfile.open(name) as f:
        f.extractall('tmp')
    os.remove(name)

    shutil.copytree('tmp/mingw64', 'mingw64')
    shutil.rmtree('tmp')


def download_libgfortran() -> None:
    source = 'http://repo.msys2.org/mingw/x86_64/'
    name = 'mingw-w64-x86_64-gcc-libgfortran-9.3.0-2-any.pkg.tar.xz'
    r = requests.get(source + name)
    with open(name, 'wb') as f:
        f.write(r.content)
    with tarfile.open(name) as f:
        f.extractall('tmp')
    os.remove(name)

    shutil.copy('tmp/mingw64/bin/libgfortran-5.dll', 'mingw64/bin/')
    shutil.rmtree('tmp')


def copy_bins(dst_path: str) -> None:
    shutil.copy('mingw64/bin/libgfortran-5.dll', dst_path)
    shutil.copy('mingw64/bin/libopenblas.dll', dst_path)
    shutil.copy('libquadmath-0.dll', dst_path)


mkdir('x64')
mkdir('x64/cpuOnly')
mkdir('x64/Debug')
mkdir('x64/Release')

print('Beginning file download with requests')
download_openblas()
download_libgfortran()
copy_bins('x64/cpuOnly/')
copy_bins('x64/Debug/')
copy_bins('x64/Release/')
