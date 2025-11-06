cmake_minimum_required(VERSION 3.10)

project(rknn_helsinki_demo)

if (ENABLE_ASAN)
    message(STATUS "BUILD WITH ADDRESS SANITIZER")
    set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
    set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
    set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
endif ()

# Add 3rdparty
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/ 3rdparty.out)

set(CMAKE_INSTALL_RPATH "$ORIGIN/lib")

# Set C++11 standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ========================================
# FetchContent for SentencePiece
# ========================================
include(FetchContent)

# 必须在 FetchContent_Declare 之前设置这些选项
set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)
set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "" FORCE)
set(SPM_USE_BUILTIN_PROTOBUF ON CACHE BOOL "" FORCE)
set(SPM_ENABLE_NFKC_COMPILE OFF CACHE BOOL "" FORCE)
set(SPM_BUILD_TEST OFF CACHE BOOL "" FORCE)

# 关键：完全禁用工具和可执行文件构建
set(SPM_ENABLE_CLI OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    sentencepiece
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../sentencepiece
)

message(STATUS "Configuring SentencePiece...")
FetchContent_MakeAvailable(sentencepiece)
message(STATUS "SentencePiece ready")

# ========================================
# Helsinki API Library (核心库)
# ========================================
set(HELSINKI_API_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/helsinki_api.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/helsinki.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/sp_tokenizer.cc
)

# 创建静态库（供可执行文件和JNI库使用）
add_library(helsinki_api_static STATIC ${HELSINKI_API_SRCS})

target_include_directories(helsinki_api_static PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBRKNNRT_INCLUDES}
    ${sentencepiece_SOURCE_DIR}/src
)

target_link_libraries(helsinki_api_static
    sentencepiece-static
    ${LIBRKNNRT}
    dl
)

# Android 平台特殊处理
if (CMAKE_SYSTEM_NAME STREQUAL "Android")
    target_link_libraries(helsinki_api_static log)
else()
    target_link_libraries(helsinki_api_static pthread)
endif()

# ========================================
# Test Executable (测试程序)
# ========================================
add_executable(${PROJECT_NAME}_test
    ${CMAKE_CURRENT_SOURCE_DIR}/main_test.cc
)

target_include_directories(${PROJECT_NAME}_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(${PROJECT_NAME}_test
    helsinki_api_static
)

# ========================================
# JNI Shared Library (供Java调用的.so库)
# ========================================
set(JNI_LIB_TARGET ${PROJECT_NAME}_jni)

add_library(${JNI_LIB_TARGET} SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/helsinki_jni.cc
)

# 设置输出名称为 librknn_helsinki_demo.so
set_target_properties(${JNI_LIB_TARGET} PROPERTIES
    OUTPUT_NAME ${PROJECT_NAME}
)

target_include_directories(${JNI_LIB_TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(${JNI_LIB_TARGET}
    helsinki_api_static
)

# ========================================
# Install
# ========================================

# 安装测试程序
install(TARGETS ${PROJECT_NAME}_test DESTINATION .)

# 安装JNI库到lib目录
install(TARGETS ${JNI_LIB_TARGET} LIBRARY DESTINATION lib)

# 安装头文件（供其他项目使用）
install(FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/helsinki_api.h
    DESTINATION include
)

# 安装模型文件
file(GLOB RKNN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/*.rknn")
install(FILES ${RKNN_FILES} DESTINATION model)

# 安装tokenizer文件
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../tokenizer_files 
        DESTINATION .
        FILES_MATCHING
        PATTERN "*.spm"
        PATTERN "*.txt")

# ========================================
# Print build information
# ========================================
message(STATUS "=================================")
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "Target SOC: ${TARGET_SOC}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "System: ${CMAKE_SYSTEM_NAME}")
message(STATUS "")
message(STATUS "Build Targets:")
message(STATUS "  - helsinki_api_static (static library)")
message(STATUS "  - ${PROJECT_NAME}_test (test executable)")
message(STATUS "  - ${JNI_LIB_TARGET} (JNI shared library)")
message(STATUS "=================================")