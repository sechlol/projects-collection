SET(IO_FILES
        FastaString.h
        FastaString.cpp
        IOUtils.cpp
        IOUtils.h)


add_library(io ${IO_FILES})
target_include_directories(io PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
