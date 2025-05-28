import os
import glob
from pkg_resources import DistributionNotFound, get_distribution, parse_version
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

op_files = glob.glob("*.cu") + glob.glob("*.cpp")
include_dirs = [os.path.abspath('include')]

setup(
    name='amp',
    version='2025',
    ext_modules=[
        CUDAExtension(
            name='cpp_pkgs._ext',
            sources=op_files,
            # sources=[
            #     "/".join(__file__.split("/")[:-1] + ["scatter_points_cuda.cu"]),
            #     "/".join(__file__.split("/")[:-1] + ["scatter_points.cpp"]),
            #     "/".join(__file__.split("/")[:-1] + ["voxelization_cuda.cu"]),
            #     "/".join(__file__.split("/")[:-1] + ["voxelization.cpp"]),
            #     "/".join(__file__.split("/")[:-1] + ["cudabind.cpp"]),
            #     "/".join(__file__.split("/")[:-1] + ["pybind.cpp"]),

            # ],
            include_dirs=include_dirs,
            ),
    ],
    cmdclass={'build_ext': BuildExtension},
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Utilities',
    ],
    python_requires='>=3.7',
    zip_safe=False)