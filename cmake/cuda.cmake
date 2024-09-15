add_compile_definitions(FZGPU_USE_CUDA)

find_package(CUDAToolkit REQUIRED)

include(GNUInstallDirs)
include(CTest)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/fzgpu_version.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/fzgpu_version.h)

add_library(fzgpu_cu_compile_settings INTERFACE)
add_library(FZGPU::cu_compile_settings ALIAS fzgpu_cu_compile_settings)

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
)

# FUNC={core,api}, BACKEND={serial,cuda,...}
add_library(fzgpu_cu_compat
    src/claunch_cuda.cu
    src/fz.cu
)
target_link_libraries(fzgpu_cu_compat
    PUBLIC FZGPU::cu_compile_settings CUDA::cudart
)
add_library(FZGPU::cu_compat ALIAS fzgpu_cu_compat)

add_library(fzgpu_cu_module
    src/fz_module.cu
)
target_link_libraries(fzgpu_cu_module
    PUBLIC FZGPU::cu_compile_settings
)
add_library(FZGPU::cu_module ALIAS fzgpu_cu_module)

add_library(fzgpu_cu_driver
    src/fz_driver.cc
    src/fz_utils.cc
)
target_link_libraries(fzgpu_cu_driver
    PUBLIC FZGPU::cu_compile_settings
    CUDA::cudart
)
add_library(FZGPU::cu_driver ALIAS fzgpu_cu_driver)

add_library(fzgpu_cu_demo
    src/fz_demo.cc
)
target_link_libraries(fzgpu_cu_demo
    PUBLIC FZGPU::cu_compile_settings
    CUDA::cudart
    FZGPU::cu_module
    FZGPU::cu_driver
)
add_library(FZGPU::cu_demo ALIAS fzgpu_cu_demo)

add_executable(fzcli src/fz_cli.cc)
set_source_files_properties(src/fz_cli.cc PROPERTIES LANGUAGE CUDA)
target_link_libraries(fzcli PRIVATE FZGPU::cu_driver FZGPU::cu_module FZGPU::cu_demo)
set_target_properties(fzcli PROPERTIES OUTPUT_NAME fz)

# enable examples and testing
if(FZGPU_BUILD_EXAMPLES)
    # add_subdirectory(example)
endif()

if(BUILD_TESTING)
    # add_subdirectory(test)
endif()

# installation
install(TARGETS
    fzgpu_cu_compile_settings
    fzgpu_cu_compat
    fzgpu_cu_module
    fzgpu_cu_driver
    EXPORT FZGPUTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# install the executable
install(TARGETS fzcli
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# export the targets to a script
install(EXPORT FZGPUTargets
    FILE FZGPUTargets.cmake
    NAMESPACE FZGPU::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FZGPU
)

# generate and install package configuration files
include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/FZGPUConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FZGPUConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/FZGPUConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FZGPU
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/FZGPUConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/FZGPUConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FZGPU
)