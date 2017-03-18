#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <string>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include "global.h"
#include "exception.h"
#include "neuralnetwork.h"
#include "backpropagation.h"
#include "trainer.h"
#include "dataset.h"
#include "nntraining.h"
#include "nntest.h"

typedef Global::uint uint;
typedef Global::real real;

void print(const std::vector<real>& v) {
  for (uint i = 0; i < v.size(); ++i)
    std::cout <<v[i] <<" ";
  return;
}

void printDataset(Dataset& ds) {
  for (uint i = 0; i < ds.getSize(); ++i) {
    std::cout <<ds.at(i).id <<" ";
    print(ds.at(i).input);
    print(ds.at(i).output);
    std::cout <<std::endl;
  }
  return;
}

void printTrSet(Dataset& ds) {
  for (uint i = 0; i < ds.getTrSetSize(); ++i) {
    std::cout <<ds.trAt(i).id <<" ";
    print(ds.trAt(i).input);
    print(ds.trAt(i).output);
    std::cout <<std::endl;
  }
  return;
}

void printVaSet(Dataset& ds) {
  for (uint i = 0; i < ds.getVaSetSize(); ++i) {
    std::cout <<ds.vaAt(i).id <<" ";
    print(ds.vaAt(i).input);
    print(ds.vaAt(i).output);
    std::cout <<std::endl;
  }
  return;
}

/**
 * Function main
 *
 * argv[0]: nome eseguibile
 * argv[1]: dataset
 * argv[2]: rseed
 * argv[3]: n
 * argv[4]: k
 * argv[5]: vuoto
 * argv[6]: vuoto
 * argv[7]: vuoto
 */
int main(int argc, char **argv) {
  uint n = Global::toUint(std::string(argv[3]));
  uint k = Global::toUint(std::string(argv[4]));

  Global::setRandSeed(Global::toUint(std::string(argv[2])));

  Dataset ds;
  std::cout <<"==== Dataset load ====" <<std::endl;
  ds.load(std::string(argv[1]),17,2);
  std::cout <<"size: " <<ds.getSize() <<std::endl;
  printDataset(ds);
  std::cout <<std::endl;

  ds.setFolds(n);
  ds.setValidationFold(k);
  for (uint i = 0; i < 5; ++i) {
    std::cout <<"==== Dataset folds ====" <<std::endl;
    ds.randomShuffleTrainingSet();
    std::cout <<"folds: " <<ds.getFolds() <<std::endl;
    for (uint i = 0; i < ds.getFolds(); ++i)
      std::cout <<"F" <<i <<".size = " <<ds.getFoldSize(i) <<", ";
    std::cout <<std::endl;
    std::cout <<"training set size: " <<ds.getTrSetSize() <<std::endl;
    printTrSet(ds);
    std::cout <<std::endl;
    std::cout <<"validation set size: " <<ds.getVaSetSize() <<std::endl;
    printVaSet(ds);
    std::cout <<std::endl;
  }

  return 0;
} // End function main
