set(CMAKE_POSITION_INDEPENDENT_CODE ON)
find_package(Boost REQUIRED python)

include_directories(..)
include_directories(include)

set (SOURCES
    tokens
    converters
    embeddings
)

set (HEADERS
    tokens
    embeddings
)

# Modify SOURCES and HEADERS to add prefix and sufix
list(TRANSFORM HEADERS PREPEND include/)
list(TRANSFORM SOURCES PREPEND src/)
list(TRANSFORM HEADERS APPEND .hpp)
list(TRANSFORM SOURCES APPEND .cpp)

include_directories(${BOOST_INCLUDE_DIRS})
message("boost lib: ${Boost_LIBRARIES}")
message("boost inc:${Boost_INCLUDE_DIR}")
add_library(pyllamacpp SHARED module.cpp ${HEADERS} ${SOURCES})

target_link_libraries(pyllamacpp llama ggml common ${Boost_LIBRARIES})
target_include_directories(pyllamacpp PUBLIC /usr/include/python3.11)

add_custom_command(TARGET pyllamacpp POST_BUILD
    COMMAND printf "Python module built, copying back..."
    COMMAND cp libpyllamacpp.so ${CMAKE_SOURCE_DIR}/pyllamacpp/model_interfaces
)

