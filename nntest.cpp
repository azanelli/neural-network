#include "nntest.h"

#include <iostream>
#include <fstream>
#include <string>
#include "global.h"
#include "exception.h"
#include "tester.h"
#include "neuralnetwork.h"

// ======================
// PRIVATE STATIC MEMBERS
// ======================

Tester* NNTest::ts;
NeuralNetwork* NNTest::nn;
bool NNTest::output;
std::string NNTest::nnfile, NNTest::dsfile, NNTest::tssave;
real NNTest::threshold;

// =====================
// PUBLIC STATIC METHODS
// =====================

/**
 * Method exec
 *
 * Esegue i seguenti passi:
 *   - Controlla i parametri globali necessari
 *   - Carica la rete neurale da file
 *   - Stampa le caratteristiche della rete neurale
 *   - Costruisce e avvia il test (salvando le risposte se richiesto)
 *   - Stampa i risultati del test
 * Restituisce 0 se l'esecuzione e` avvenuta correttamente, un numero diverso
 * da 0 se ci sono stati errori (per esempio mancano dei parametri).
 */
int NNTest::exec() {
  // Legge e controlla i parametri
  if (!checkParameters()) return -1;

  // Carica la rete neurale dal file
  nn = new NeuralNetwork();
  std::ifstream ifs(nnfile.c_str());
  if (!ifs.is_open()) throw file_error("In NNTest::exec");
  ifs >>(*nn);

  // Stampa le caratteristiche della rete neurale caricata
  printNeuralNetworkInfo();

  // Costruisce il test passandogli i parametri
  ts = new Tester(nn, output);
  ts->setDataSet(dsfile);
  if (!tssave.empty()) ts->setSaveModelResponses(tssave);
  ts->setThreshold(threshold);

  // Avvia il test
  ts->start();

  // Stampa i risultati del test
  printTestInfo();

  // Elimina le strutture utilizzate e termina
  delete nn;
  delete ts;
  return 0;
} // End method exec

// ======================
// PRIVATE STATIC METHODS
// ======================

/**
 * Method checkParameters
 *
 * Controlla i parametri, verificando che esistano quelli obbligatori, che i
 * valori abbiano senso, ed assegnando un valore ad ogni variabile che
 * corrisponde ad un parametro. In caso di errore sui parametri restituisce
 * false.
 */
bool NNTest::checkParameters ( ) {
  std::vector<std::string> required;
  std::vector<std::string> missingarg;
  std::string strunits;
  // --nnfile
  if (Global::getParam("nnfile").empty())
    required.push_back("--nnfile");
  else if (Global::getParam("nnfile") == "nnfile")
    missingarg.push_back("--nnfile");
  else nnfile = Global::getParam("nnfile");
  // --dsfile
  if (Global::getParam("dsfile").empty())
    required.push_back("--dsfile");
  else if (Global::getParam("dsfile") == "dsfile")
    missingarg.push_back("--dsfile");
  else dsfile = Global::getParam("dsfile");
  // --output
  if (Global::getParam("output").empty())
    output = false; // valore di default
  else output = true;
  // --tssave
  if (Global::getParam("tssave").empty())
    tssave = ""; // valore di default
  else if (Global::getParam("tssave") == "tssave")
    missingarg.push_back("--tssave");
  else tssave = Global::getParam("tssave");
  // --threshold
  if (Global::getParam("threshold").empty())
    threshold = 0.5; // valore di default
  else if (Global::getParam("threshold") == "threshold")
    missingarg.push_back("--threshold");
  else threshold = Global::toReal(Global::getParam("threshold"));
  // controlla se ci sono errori
  if (!required.empty()) {
    std::cout <<"The follow parameters are required (in test mode)";
    std::cout <<std::endl;
    for (std::size_t i = 0; i < required.size(); ++i)
      std::cout <<"  " <<required[i] <<std::endl;
    return false;
  }
  if (!missingarg.empty()) {
    std::cout <<"The follow parameters requires an argument (in test mode)";
    std::cout <<std::endl;
    for (std::size_t i = 0; i < missingarg.size(); ++i)
      std::cout <<"  " <<missingarg[i] <<std::endl;
    return false;
  }
  // verifica i valori dei parametri
  if (threshold < 0 || threshold > 1) {
    std::cout <<"The parameter --tolerance must be in the range [0,1]";
    std::cout <<std::endl;
    return false;
  }
  return true;
} // End method checkParameters

/**
 * Method printNeuralNetworkInfo
 *
 * Stampa su standard output tutte le informazioni relative alla rete neurale.
 */
void NNTest::printNeuralNetworkInfo() {
  std::cout <<"# neural network" <<std::endl;
  std::cout <<"inputs: " <<nn->getNumberOfInputs() <<"\n";
  std::cout <<"outputs: " <<nn->getNumberOfOutputs() <<"\n";
  std::cout <<"hidden layers: " <<nn->getNumberOfHiddenLayers() <<"\n";
  std::cout <<"units in any layer:";
  for (uint i = 0; i < nn->getNumberOfLayers(); ++i)
    std::cout <<" " <<nn->getNumberOfUnits(i);
  std::cout <<" (total " <<nn->getNumberOfUnits()  <<")\n";
  return;
} // End of method printNeuralNetworkInfo

/**
 * Method printTestInfo
 *
 * Stampa su standard output tutte le informazioni relative al test eseguito.
 */
void NNTest::printTestInfo () {
  std::cout <<"# test results" <<std::endl;
  std::cout <<"dataset size: " <<ts->getDatasetDimension() <<"\n";
  if (!output) {
    std::cout <<"results write on: " <<tssave <<"\n";
    return;
  }
  std::cout <<"hits: " <<ts->getNumberOfHits() <<"\n";
  std::cout <<"missed: " <<ts->getNumberOfMissed() <<"\n";
  std::cout <<"accuracy: " <<ts->getAccuracy() <<"% \n";
  std::cout <<"quadratic mean error: " <<ts->getQuadraticError() <<"\n";
  return;
} // End of method printTestInfo
