project('fedavg_mpi', 'c')
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
  'fedavg_mpi', 'main.c', 'cythoned/library.c',
  dependencies: [mpi, python],
  include_directories: numpy_header
)
