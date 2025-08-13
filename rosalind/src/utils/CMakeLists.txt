SET(UTILS_FILES
        CodonTable.h
        ContainerUtils.h
        MassTable.h
        RunUtils.h
        RunUtils.cpp
        StringUtils.h
        StringUtils.cpp
        VectorView.h
        LCS/SuffixMap.cpp
        LCS/SuffixMap.h
        LCS/LCS.cpp
        LCS/LCS.h
        LCS/SlidingWindow.cpp
        LCS/SlidingWindow.h LCS/StringCompactor.h LCS/StringCompactor.cpp)

add_library(utils STATIC ${UTILS_FILES})
target_include_directories(utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})