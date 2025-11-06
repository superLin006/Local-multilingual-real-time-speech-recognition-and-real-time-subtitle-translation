cmake_minimum_required(VERSION 3.10)

project(rknn_whisper_demo)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

if (ENABLE_ASAN)
	message(STATUS "BUILD WITH ADDRESS SANITIZER")
	set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
endif ()

set(rknpu_whisper_file rknpu2/whisper.cc)
if(TARGET_SOC STREQUAL "rk1808" OR TARGET_SOC STREQUAL "rv1109" OR TARGET_SOC STREQUAL "rv1126")
    set(rknpu_whisper_file rknpu1/whisper.cc)
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/ 3rdparty.out)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../utils/ utils.out)

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(STATUS "64bit")
    set (TARGET_LIB_ARCH lib64)
else()
    message(STATUS "32bit")
    set (TARGET_LIB_ARCH lib)
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Android")
  set(OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/opencv/opencv-android-sdk-build/sdk/native/jni/abi-${CMAKE_ANDROID_ARCH_ABI})
else()
    if(TARGET_LIB_ARCH STREQUAL "lib")
      set(OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/opencv/opencv-linux-armhf/share/OpenCV)
    else()
      set(OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/opencv/opencv-linux-aarch64/share/OpenCV)
    endif()
  endif()
find_package(OpenCV REQUIRED)

set(CMAKE_INSTALL_RPATH "$ORIGIN/lib")

add_executable(${PROJECT_NAME}
    main.cc
    process.cc
    streaming_manager.cc
    mock_streaming_feeder.cc
    thread_safe_buffer.cc
    result_queue.cc
    ${rknpu_whisper_file}
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBRKNNRT_INCLUDES}
    ${LIBFFTW_INCLUDES}
    ${LIBTIMER_INCLUDES}
)

target_link_libraries(${PROJECT_NAME}
    fileutils
    audioutils
    ${LIBRKNNRT}
    ${LIBFFTW}
    ${OpenCV_LIBS}
    $<$<PLATFORM_ID:Android>:log>
    $<$<PLATFORM_ID:Linux>:Threads::Threads>
)

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION .)

set(JNI_LIB_TARGET ${PROJECT_NAME}_jni)

add_library(${JNI_LIB_TARGET} SHARED
    whisper_jni_wrapper.cc
    main.cc
    process.cc
    streaming_manager.cc
    mock_streaming_feeder.cc
    thread_safe_buffer.cc
    result_queue.cc
    ${rknpu_whisper_file}
)

set_target_properties(${JNI_LIB_TARGET} PROPERTIES
    OUTPUT_NAME ${PROJECT_NAME}
)

target_include_directories(${JNI_LIB_TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBRKNNRT_INCLUDES}
    ${LIBFFTW_INCLUDES}
    ${LIBTIMER_INCLUDES}
)

target_link_libraries(${JNI_LIB_TARGET}
    fileutils
    audioutils
    ${LIBRKNNRT}
    ${LIBFFTW}
    ${OpenCV_LIBS}
    $<$<PLATFORM_ID:Android>:log>
    $<$<PLATFORM_ID:Linux>:Threads::Threads>
)

install(TARGETS ${JNI_LIB_TARGET} LIBRARY DESTINATION lib)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../model/test_en.wav DESTINATION ./model)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../model/test_zh.wav DESTINATION ./model)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../model/test_ja.wav DESTINATION ./model)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../model/test_de.wav DESTINATION ./model)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../model/test_fr.wav DESTINATION ./model)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../model/test_es.wav DESTINATION ./model)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../model/vocab.txt DESTINATION ./model)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../model/mel_80_filters.txt DESTINATION ./model)
file(GLOB RKNN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/*.rknn")
install(FILES ${RKNN_FILES} DESTINATION model)