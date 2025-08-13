SET(EXERCISE_FILES
        Result.h
        ConsensusAndProfile.h
        ConsensusAndProfile.cpp
        Exercise.h
        Exercise.cpp
        ExerciseImpl.h
        Ex1Dna.h
        Ex1Dna.cpp
        Ex2Rna.cpp
        Ex2Rna.h
        Ex3Revc.cpp
        Ex3Revc.h
        Ex4Hamm.cpp
        Ex4Hamm.h
        Ex5Gc.cpp
        Ex5Gc.h
        Ex6Fib.cpp
        Ex6Fib.h
        Ex7Fibd.cpp
        Ex7Fibd.h
        Ex8Iprb.cpp
        Ex8Iprb.h
        Ex9Prot.cpp
        Ex9Prot.h
        ExA0Subs.cpp
        ExA0Subs.h
        ExA1Cons.cpp
        ExA1Cons.h
        ExA2Grph.cpp
        ExA2Grph.h
        ExA3Mprt.cpp
        ExA3Mprt.h
        ExA4Iev.cpp
        ExA4Iev.h
        ExA5Mrna.cpp
        ExA5Mrna.h
        ExA6Prtm.cpp
        ExA6Prtm.h
        ExA7Lcsm.cpp
        ExA7Lcsm.h)

add_library(exercises ${EXERCISE_FILES})
target_include_directories(exercises PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(exercises PUBLIC io utils)