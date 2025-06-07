cmake_minimum_required(VERSION 3.10)
project(torch_ppa_and_npu_sim LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -- Libtorch (C++ API) --
# adjust this to wherever you unpacked libtorch
set(Torch_DIR $ENV{HOME}/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)

# -- OpenCV (for image loading) --
find_package(OpenCV REQUIRED)

# -- NVML (for power & util) --
find_library(NVML_LIBRARY nvml HINTS /usr/lib /usr/lib/x86_64-linux-gnu)
find_path  (NVML_INCLUDE_DIR nvml.h HINTS /usr/include /usr/local/include)

# -- CUDA runtime (for cudaDeviceSynchronize, etc) --
find_package(CUDA REQUIRED)

include_directories(
  ${TORCH_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${NVML_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIRS}
)

# ---- GPU PPA tool ----
add_executable(gpu_ppa main.cpp)
target_link_libraries(gpu_ppa
    PRIVATE
      ${TORCH_LIBRARIES}
      ${OpenCV_LIBS}
      ${NVML_LIBRARY}
)

# ---- NPU-simulator tool ----
add_executable(npu_sim main_npu_sim.cpp)
target_link_libraries(npu_sim
    PRIVATE
      ${TORCH_LIBRARIES}
      ${OpenCV_LIBS}
      ${NVML_LIBRARY}
      ${CUDA_CUDA_LIBRARY}   # for cudaDeviceSynchronize, etc
)

# hide symbols for libtorch
set_property(TARGET gpu_ppa PROPERTY CXX_VISIBILITY_PRESET hidden)
set_property(TARGET gpu_ppa PROPERTY VISIBILITY_INLINES_HIDDEN ON)
set_property(TARGET npu_sim PROPERTY CXX_VISIBILITY_PRESET hidden)
set_property(TARGET npu_sim PROPERTY VISIBILITY_INLINES_HIDDEN ON)
