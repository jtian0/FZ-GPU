add_compile_definitions(FZGPU_USE_CUDA)

find_package(CUDAToolkit REQUIRED)

include(GNUInstallDirs)
include(CTest)

# configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/fzgpu_version.h.in
# ${CMAKE_CURRENT_BINARY_DIR}/include/fzgpu_version.h)
add_library(fzgpu_cu_compile_settings INTERFACE)

target_compile_definitions(
    fzgpu_cu_compile_settings
    INTERFACE $<$<COMPILE_LANG_AND_ID:CUDA,Clang>:__STRICT_ANSI__>)
target_compile_options(
    fzgpu_cu_compile_settings
    INTERFACE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda
    --expt-relaxed-constexpr -Wno-deprecated-declarations>)
target_compile_features(fzgpu_cu_compile_settings INTERFACE cxx_std_17 cuda_std_17)

target_include_directories(
    fzgpu_cu_compile_settings
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>

    # $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/cusz>)
)

# FUNC={core,api}, BACKEND={serial,cuda,...}
add_library(fzgpu_cu_compat
    src/claunch_cuda.cu
    src/fz.cu
)
target_link_libraries(fzgpu_cu_compat
    PUBLIC fzgpu_cu_compile_settings CUDA::cudart
)

add_library(fzgpu_cu_module
    src/fz_module.cu
)
target_link_libraries(fzgpu_cu_module
    PUBLIC fzgpu_cu_compile_settings
)

add_library(fzgpu_cu_driver
    src/fz_driver.cc
)
target_link_libraries(fzgpu_cu_driver
    PUBLIC fzgpu_cu_compile_settings
    CUDA::cudart
)

add_executable(fzcli src/fz_cli.cc)
set_source_files_properties(src/fz_cli.cc PROPERTIES LANGUAGE CUDA)
target_link_libraries(fzcli PRIVATE fzgpu_cu_driver fzgpu_cu_driver fzgpu_cu_module)
set_target_properties(fzcli PROPERTIES OUTPUT_NAME fz)

# enable examples and testing
if(FZGPU_BUILD_EXAMPLES)
    # add_subdirectory(example)
endif()

if(BUILD_TESTING)
    # add_subdirectory(test)
endif()

# installation
install(TARGETS fzgpu_cu_compile_settings EXPORT CUSZTargets)
install(TARGETS fzgpu_cu_compat EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

# install(TARGETS cusz EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
# install(TARGETS cusz-bin EXPORT CUSZTargets)
