#include <iostream>
#include <fstream>
#include "global.h"
#include "nntraining.h"
#include "nntest.h"

// Dichiarazione di funzioni
bool checkParameters();
void printHelp();

// Variabili globali
enum Mode { training, test } mode;
uint rseed;

/**
 * Function main
 *
 * Funzione iniziale dell'applicazione. Legge i parametri passati al programma
 * inserendoli nella classe Global, inizializza il generatore di numeri casuali
 * con il seme passato come parametro e infine avvia l'esecuzione della
 * modalita` richiesta. Le modalita` possono essere "training" oppure "test".
 * Per le informazioni sul programma, i parametri e le modalita` di esecuzione
 * si puo` avviare l'applicazione con il parametro --help.
 */
int main(int argc, char **argv) {
  // Legge i parametri passati al programma
  Global::readParameters(argc,argv);

  // Controlla i parametri passati
  if (!checkParameters()) return -1;

  // Inizializza il generatore di numeri casuali
  if (rseed != 0) Global::setRandSeed(rseed);
  else Global::setRandSeed(time(NULL)%10000);

  // Esegue la modalita` scelta
  switch (mode) {
  case training :
    return NNTraining::exec();
  case test :
    return NNTest::exec();
  } // end switch

  return 0;
} // End function main

/**
 * Function checkParameters
 *
 * Controlla i parametri globali, in particolare:
 *   --help  : visualizza l'help del programma ed esce
 *   --mode  : controlla che la modalita` scelta sia valida
 *   --rseed : imposta il seme per il generatore di numeri casuali
 * Se i parametri sono validi (e non e` stato richiesto l'help) la funzione
 * restituisce true e le variabili globali contengono i valori impostati,
 * altrimenti restituisce false.
 */
bool checkParameters() {
  // parametro --help
  if (!Global::getParam("help").empty() || !Global::getParam("h").empty()) {
    printHelp();
    return false;
  }
  // parametro --mode
  if (Global::getParam("mode").empty()) {
    std::cout <<"Option --mode is required (try with --help)." <<std::endl;
    return false;
  }
  std::string strmode = Global::getParam("mode");
  if (strmode == "training") {
    mode = training;
  } else if (strmode == "test") {
    mode = test;
  } else if (strmode == "mode") {
    std::cout <<"Option --mode requires an argument (try with --help)";
    std::cout <<std::endl;
    return false;
  } else {
    std::cout <<"Mode \"" <<strmode <<"\" is not valid (try with --help)";
    std::cout <<std::endl;
    return false;
  }
  // parametro --rseed
  if (Global::getParam("rseed").empty()) {
    rseed = 0;
  } else if (Global::getParam("rseed") == "rseed") {
    std::cout <<"Option --rseed requires an argument" <<std::endl;
    return false;
  } else {
    rseed = Global::toUint(Global::getParam("rseed"));
  }
  return true;
} // End function checkParameters

/**
 * Function printHelp
 *
 * Stampa su standard output l'help del programma letto dal file help.txt.
 */
void printHelp() {
  std::string line;
  std::ifstream file("help.txt");
  if ( file.is_open() ) {
    while ( file.good() ) {
      std::getline(file, line);
      std::cout <<line <<std::endl;
    }
  } else {
    std::cout <<"Error in reading help." <<std::endl;
  }
  return;
} // End function printHelp
