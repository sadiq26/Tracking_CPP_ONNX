# CMakeList.txt : CMake project for L2F_Tracking_V2, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("L2F_Tracking_V2")

set(OpenCV_DIR "E:/Coding_Space/New folder/L2F_tracking_V1/third_party/opencv/opencv-4.10.0/build")
find_package(OpenCV REQUIRED)


set(ONNXRUNTIME_DIR "E:/Coding_Space/New folder/L2F_tracking_V1/third_party/onnxruntime-win-x64-1.24.3")

set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_DIR}/include")
set(ONNXRUNTIME_LIB "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")







# Add source to this project's executable.
add_executable (L2F_Tracking_V2 "L2F_Tracking_V2.cpp" "L2F_Tracking_V2.h" "utils.cpp" "main.cpp")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET L2F_Tracking_V2 PROPERTY CXX_STANDARD 20)
endif()

# TODO: Add tests and install targets if needed.



target_include_directories(L2F_Tracking_V2 PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIR}
)



target_link_libraries(L2F_Tracking_V2 ${OpenCV_LIBS} ${ONNXRUNTIME_LIB})
