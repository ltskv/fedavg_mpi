project('fedavg_mpi', 'c')

add_project_arguments(
  '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION',
  language: 'c'
)

compiler = meson.get_compiler('c')
if compiler.has_argument('-Wno-unused-command-line-argument')
  add_global_arguments('-Wno-unused-command-line-argument', language: 'c')
endif

if compiler.has_link_argument('-Wl,-w')
  add_link_args = ['-Wl,-w']
else
  add_link_args = []
endif

mpi = dependency('mpi')
python = dependency('python3')
numpy_header = include_directories(run_command(
  'python3', '-c', 'import numpy;print(numpy.get_include())'
).stdout().strip())

cython = find_program('cython')
bridge = custom_target('bridge',
  output: ['bridge.c', 'bridge.h'],
  input: 'bridge.pyx',
  command: [cython, '-3', '@INPUT@', '-o', 'bridge.c'])

executable('fedavg_mpi',
  'main.c', bridge,
  dependencies: [mpi, python],
  include_directories: numpy_header,
  link_args: add_link_args)
