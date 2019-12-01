project('fedavg_mpi', 'c')
add_global_arguments('-Wno-unused-command-line-argument', language: 'c')
add_project_arguments(
  '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION',
  language: 'c'
)

mpi = dependency('mpi')
python = dependency('python3')
numpy_header = include_directories(run_command(
  'python3', '-c', 'import numpy;print(numpy.get_include())'
).stdout().strip())

executable(
  'fedavg_mpi', 'main.c', 'cythoned/bridge.c',
  dependencies: [mpi, python],
  include_directories: numpy_header,
  link_args: '-Wl,-w'
)
