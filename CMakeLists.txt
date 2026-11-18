cmake_minimum_required(VERSION 3.22)
project(prism_rp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(PRISM_RP_VALIDATE_SHADERS "Validate GLSL 4.60 shaders via glslangValidator during build" ON)
set(GODOT_TARGET "template_debug" CACHE STRING "Godot build target (template_debug or template_release)")

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(PRISM_PLATFORM "linux")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(PRISM_PLATFORM "windows")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(PRISM_PLATFORM "macos")
else()
    message(FATAL_ERROR "Unsupported target platform: ${CMAKE_SYSTEM_NAME}")
endif()

if(PRISM_PLATFORM STREQUAL "macos")
    set(PRISM_ARCH "universal")
    set(CMAKE_OSX_ARCHITECTURES "x86_64;arm64" CACHE STRING "" FORCE)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64|arm64")
    set(PRISM_ARCH "arm64")
else()
    set(PRISM_ARCH "x86_64")
endif()

# godot-cpp picks debug/release features from GODOTCPP_TARGET; keep it in sync with
# GODOT_TARGET so the binary's features match its file name.
set(GODOTCPP_TARGET "${GODOT_TARGET}" CACHE STRING "" FORCE)
# Godot API the extension is built against (godot-cpp v10 supports several; keep in sync with
# SConstruct and compatibility_minimum in prism_rp.gdextension).
set(GODOTCPP_API_VERSION "4.7" CACHE STRING "" FORCE)
add_subdirectory(thirdparty/godot-cpp SYSTEM)

file(GLOB_RECURSE PRISM_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/addons/prism_rp/src/*.cpp"
)
file(GLOB MESHOPTIMIZER_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/meshoptimizer/src/*.cpp")
list(APPEND PRISM_SOURCES ${MESHOPTIMIZER_SOURCES})

add_library(${PROJECT_NAME} SHARED ${PRISM_SOURCES})

target_include_directories(${PROJECT_NAME}
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/addons/prism_rp/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/renderdoc"
        "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/meshoptimizer/src"
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        godot::cpp
        ${CMAKE_DL_LIBS}
)

if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /W4 /Zc:__cplusplus)
else()
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

set(PRISM_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/addons/prism_rp/bin")
set(PRISM_OUTPUT_NAME "prism_rp.${PRISM_PLATFORM}.${GODOT_TARGET}.${PRISM_ARCH}")

set_target_properties(${PROJECT_NAME} PROPERTIES
    PREFIX "lib"
    OUTPUT_NAME "${PRISM_OUTPUT_NAME}"
    LIBRARY_OUTPUT_DIRECTORY "${PRISM_OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${PRISM_OUTPUT_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${PRISM_OUTPUT_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${PRISM_OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${PRISM_OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${PRISM_OUTPUT_DIR}"
)

if(PRISM_RP_VALIDATE_SHADERS)
    find_program(GLSLANG_VALIDATOR glslangValidator HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin")
    if(NOT GLSLANG_VALIDATOR)
        message(FATAL_ERROR "glslangValidator not found. Install the Vulkan SDK or glslang-tools, or set -DPRISM_RP_VALIDATE_SHADERS=OFF.")
    endif()

    file(GLOB_RECURSE PRISM_SHADERS CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/addons/prism_rp/shaders/*.glsl"
    )
    list(FILTER PRISM_SHADERS EXCLUDE REGEX ".*/include/.*")

    set(SPIRV_STAMP_FILES "")
    set(SPIRV_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/spirv_check")
    file(MAKE_DIRECTORY "${SPIRV_OUT_DIR}")

    foreach(SHADER_FILE ${PRISM_SHADERS})
        get_filename_component(SHADER_NAME "${SHADER_FILE}" NAME)
        if(SHADER_NAME MATCHES "\\.comp\\.glsl$")
            set(SHADER_STAGE "comp")
        elseif(SHADER_NAME MATCHES "\\.vert\\.glsl$")
            set(SHADER_STAGE "vert")
        elseif(SHADER_NAME MATCHES "\\.frag\\.glsl$")
            set(SHADER_STAGE "frag")
        elseif(SHADER_NAME MATCHES "\\.mesh\\.glsl$")
            set(SHADER_STAGE "mesh")
        elseif(SHADER_NAME MATCHES "\\.task\\.glsl$")
            set(SHADER_STAGE "task")
        else()
            message(FATAL_ERROR "Shader ${SHADER_FILE} must end with .<stage>.glsl (e.g., .comp.glsl)")
        endif()

        set(SPV_OUT "${SPIRV_OUT_DIR}/${SHADER_NAME}.spv")
        add_custom_command(
            OUTPUT "${SPV_OUT}"
            COMMAND "${GLSLANG_VALIDATOR}"
                -V
                --target-env vulkan1.3
                -S "${SHADER_STAGE}"
                "-I${CMAKE_CURRENT_SOURCE_DIR}/addons/prism_rp/shaders/include"
                "${SHADER_FILE}"
                -o "${SPV_OUT}"
            DEPENDS "${SHADER_FILE}"
            COMMENT "[Prism RP] Validating Vulkan 1.3 SPIR-V: ${SHADER_NAME}"
            VERBATIM
        )
        list(APPEND SPIRV_STAMP_FILES "${SPV_OUT}")
    endforeach()

    add_custom_target(prism_rp_validate_shaders DEPENDS ${SPIRV_STAMP_FILES})
    add_dependencies(${PROJECT_NAME} prism_rp_validate_shaders)
endif()
