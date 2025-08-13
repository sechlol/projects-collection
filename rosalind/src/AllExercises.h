#pragma once
#include <memory>
#include <unordered_map>
#include "exercises/Exercise.h"
#include "exercises/Ex1Dna.h"
#include "exercises/Ex2Rna.h"
#include "exercises/Ex3Revc.h"
#include "exercises/Ex4Hamm.h"
#include "exercises/Ex5Gc.h"
#include "exercises/Ex6Fib.h"
#include "exercises/Ex7Fibd.h"
#include "exercises/Ex8Iprb.h"
#include "exercises/Ex9Prot.h"
#include "exercises/ExA0Subs.h"
#include "exercises/ExA1Cons.h"
#include "exercises/ExA2Grph.h"
#include "exercises/ExA3Mprt.h"
#include "exercises/ExA4Iev.h"
#include "exercises/ExA5Mrna.h"
#include "exercises/ExA6Prtm.h"
#include "exercises/ExA7Lcsm.h"

typedef std::unordered_map<std::string, std::shared_ptr<Exercise>> ExerciseList;

ExerciseList CreateExercises(){
    return {
            {"dna", std::make_shared<Ex1Dna>()},
            {"rna", std::make_shared<Ex2Rna>()},
            {"revc", std::make_shared<Ex3Revc>()},
            {"hamm", std::make_shared<Ex4Hamm>()},
            {"gc", std::make_shared<Ex5Gc>()},
            {"fib", std::make_shared<Ex6Fib>()},
            {"fibd", std::make_shared<Ex7Fibd>()},
            {"iprb", std::make_shared<Ex8Iprb>()},
            {"prot", std::make_shared<Ex9Prot>()},
            {"subs", std::make_shared<ExA0Subs>()},
            {"cons", std::make_shared<ExA1Cons>()},
            {"grph", std::make_shared<ExA2Grph>()},
            {"mprt", std::make_shared<ExA3Mprt>()},
            {"iev", std::make_shared<ExA4Iev>()},
            {"mrna", std::make_shared<ExA5Mrna>()},
            {"prtm", std::make_shared<ExA6Prtm>()},
            {"lcsm", std::make_shared<ExA7Lcsm>()}
    };
}