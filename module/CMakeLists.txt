set(CMAKE_POSITION_INDEPENDENT_CODE ON)
find_package(Boost REQUIRED python)

include_directories(${BOOST_INCLUDE_DIRS})
message("boost lib: ${Boost_LIBRARIES}")
message("boost inc:${Boost_INCLUDE_DIR}")
add_library(pyllamacpp SHARED module.cpp)

target_link_libraries(pyllamacpp llama ggml ${Boost_LIBRARIES})
target_include_directories(pyllamacpp PUBLIC /usr/include/python3.10)

add_custom_command(TARGET pyllamacpp POST_BUILD
    COMMAND printf "Python module built, copying back..."
    COMMAND cp libpyllamacpp.so ${CMAKE_SOURCE_DIR}/pyllamacpp
)

