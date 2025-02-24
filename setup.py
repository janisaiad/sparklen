# Author: Romain E. Lacoste
# License: BSD-3-Clause

import os
import subprocess
import numpy
from setuptools import Extension, setup, find_packages

def ensure_init_files(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if '__init__.py' not in filenames:
            init_file_path = os.path.join(dirpath, '__init__.py')
            with open(init_file_path, 'w') as f:
                f.write("# init file for package\n")

def run_swig(interface_file, output_file, include_dirs):
    swig_cmd = [
        'swig',
        '-python',  # Generate Python bindings
        '-c++',     # The code is C++
        *[f'-I{d}' for d in include_dirs],  # Include directories
        '-o', output_file,  # Output file for SWIG-generated C++ wrapper code
        interface_file  # Input SWIG interface file
    ]
    
    try:
        subprocess.check_call(swig_cmd)
    except subprocess.CalledProcessError as e:
        print(f"SWIG command failed for {interface_file} with error:", e)
        raise

def create_extension(module_name, module_dir, source_files):
    # Define directory paths
    interface_dir = os.path.normpath(os.path.join('lib', 'interface', 'sparkle', module_dir))
    src_dir = os.path.normpath(os.path.join('lib', 'src', module_dir))
    build_dir = os.path.normpath(os.path.join('sparkle', module_dir, 'build'))
    
    # Define file paths
    interface_file = os.path.normpath(os.path.join(interface_dir, f'{module_name}_module.i'))
    wrapper_file = os.path.normpath(os.path.join(build_dir, f'{module_name}_wrap.cxx'))
    
    # Ensure build directory exists
    if not os.path.exists(build_dir):
        os.makedirs(build_dir)
    
    # Define include directories
    include_dirs = [
        os.path.normpath(os.path.join('lib', 'include')),
        os.path.normpath(os.path.join('lib', 'include', 'sparkle')),
        os.path.normpath(os.path.join('lib', 'interface', 'sparkle')),
        os.path.normpath(os.path.join('lib', 'interface'))
    ]
    
    # Run SWIG to generate wrapper code
    run_swig(interface_file, wrapper_file, include_dirs)
    
    # Determine compile arguments based on platform
    extra_compile_args = ['-std=c++11'] if os.name != 'nt' else ['/std:c++11']
    
    # Define source files for the extension
    sources = [wrapper_file] + [os.path.normpath(os.path.join(src_dir, src)) for src in source_files]
    
    # Return the Extension object
    return Extension(
        name=f"sparkle.{module_dir.replace('/', '.')}.build._{module_name}",
        sources=sources,
        include_dirs=include_dirs + [numpy.get_include()],
        extra_compile_args=extra_compile_args,
        language='c++'
    )
    
# Create extensions for module1 and module2 with multiple source files
array_extension = create_extension(
    module_name='array',
    module_dir='array',
    source_files=[
    ]
)

hawkes_model_extension = create_extension(
    module_name='hawkes_model',
    module_dir='hawkes/model',
    source_files=[
        'model_hawkes_exp_least_squares_single.cpp',
        'model_hawkes_exp_least_squares.cpp',
        'model_hawkes_exp_log_likelihood_single.cpp',
        'model_hawkes_exp_log_likelihood.cpp'
    ]
)

prox_extention = create_extension(
    module_name='prox',
    module_dir='prox',
    source_files=[
        'prox_zero.cpp',
        'prox_l1.cpp',
        'prox_l2.cpp',
        'prox_elastic_net.cpp'
    ]
)

ext_sparkle_modules = [
    array_extension,
    hawkes_model_extension,
    prox_extention
]

setup(
    name='Sparkle',
    version='0.1.0',
    author='Romain E. Lacoste',
    author_email='romain.lacoste@polytechnique.edu',
    url='https://github.com/romain-e-lacoste/sparkle',
    description='A statistical learning toolkit for high-dimensional Hawkes processes in Python',
    ext_modules=ext_sparkle_modules,
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'matplotlib',
        'numpy',
        'pandas',
        'scipy',
        'seaborn',
        'scikit-learn',  
        'tabulate',
        'tqdm'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Programming Language :: C++',
        'Programming Language :: Python',
        'Topic :: Software Development :: Libraries',
        'Topic :: Scientific/Engineering',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'License :: OSI Approved :: BSD License'
    ]
  )
  
ensure_init_files('sparkle')   