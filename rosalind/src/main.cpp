#include <iostream>
#include <memory>
#include "io/IOUtils.h"
#include "exercises/Exercise.h"
#include "AllExercises.h"

using namespace std;

const int TEST_ROUNDS = 10;

double AverageRunTime(const shared_ptr<Exercise>& exercise){
    double sumTimes = 0;
    for (int i = 0; i < TEST_ROUNDS; i++) {
        sumTimes += exercise->RunAndCacheInput().ExecutionTime;
        if (sumTimes > 1500){
            return sumTimes/(double)(i+1);
        }
    }
    return sumTimes/(double)TEST_ROUNDS;
}

void TestAll(const ExerciseList& list){
    cout<< "Testing " << list.size() << " exercises" << endl;
    for (const auto& entry : list)
        cout << "=> " << entry.first << "\t| " << (entry.second->Test() ? "*PASSED*" : "!FAILED!") << " | " << AverageRunTime(entry.second) << "ms" << endl;
}

void RunOne(const string& name, const shared_ptr<Exercise>& exercise){
    cout << "Running " << name << endl;

    bool passed = exercise->Test();
    auto result = exercise->Run();

    cout << "==========="<<endl;
    cout << result.Outcome<<endl;
    cout << "==========="<<endl;

    if (!passed){
        cout<<"Test FAILED."<<endl;
    }
    else{
        WriteToFile("solution.txt", string(result.Outcome));

        cout << "Executed in " << result.ExecutionTime << "ms" <<endl;
    }
}

int main(){
    auto exercises = CreateExercises();
/**/
    TestAll(exercises);
/*/
    const string TEST_TO_RUN = "lcsm";
    RunOne(TEST_TO_RUN, exercises[TEST_TO_RUN]);
/**/
    return 0;
}

