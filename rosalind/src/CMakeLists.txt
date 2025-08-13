add_executable(
        rosalind
        main.cpp
        AllExercises.h)

target_link_libraries(rosalind exercises utils io)