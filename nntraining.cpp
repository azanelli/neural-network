#include "nntraining.h"

#include <iostream>
#include <fstream>
#include <sys/time.h>
#include "global.h"
#include "exception.h"
#include "neuralnetwork.h"
#include "backpropagation.h"
#include "trainer.h"

// ======================
// PRIVATE STATIC MEMBERS
// ======================

Trainer* NNTraining::tr;
NeuralNetwork* NNTraining::nn;
BackPropagation* NNTraining::bp;
uint NNTraining::inputs, NNTraining::outputs, NNTraining::hlayers;
std::vector<uint> NNTraining::units;
real NNTraining::eta, NNTraining::alpha, NNTraining::lambda;
std::string NNTraining::trfile, NNTraining::trsave, NNTraining::nnsave;
uint NNTraining::folds, NNTraining::maxfolds;
uint NNTraining::maxepochs, NNTraining::shuffle;
real NNTraining::stoperr, NNTraining::stopacc, NNTraining::threshold;
float NNTraining::stoperrch;
uint NNTraining::stoperrchep;
timeval NNTraining::time_start, NNTraining::time_end;
clock_t NNTraining::tcpu_start, NNTraining::tcpu_end;
uint NNTraining::mepochs = 0;
real NNTraining::mtrerr = 0.0, NNTraining::mvaerr = 0.0;
real NNTraining::mtracc = 0.0, NNTraining::mvaacc = 0.0;
real NNTraining::mtrerrmin = 0.0, NNTraining::mvaerrmin = 0.0;
real NNTraining::mtraccmax = 0.0, NNTraining::mvaaccmax = 0.0;
double NNTraining::mtime = 0.0, NNTraining::mtcpu = 0.0;

// =====================
// PUBLIC STATIC METHODS
// =====================

/**
 * Method exec
 *
 * Esegue i seguenti passi:
 *   - Controlla i parametri globali necessari.
 *   - Costruisce la rete neurale secondo i parametri impostati.
 *   - Costruisce l'algoritmo di back-propagation con i parametri impostati.
 *   - Stampa in output tutte le caratteristiche della rete neurale e dell'
 *     algoritmo.
 *   - Attraverso la classe Trainer avvia il training sulla rete neurale con
 *     l'algoritmo di back-propagation.
 *   - Per ogni folds impostato stampa in output i risultati ottenuti e, se
 *     richiesto, salva su file i risultati del training e/o i modelli
 *     ottenuti dopo il training per ogni folds.
 *   - Al termine del training stampa la media dei risultati nei folds su
 *     cui si e` fatto training.
 * Restituisce 0 se l'esecuzione e` avvenuta correttamente, un numero diverso
 * da 0 se ci sono stati errori (per esempio mancano dei parametri).
 */
int NNTraining::exec() {
  // Legge e controlla i parametri
  if (!checkParameters()) return -1;

  // Stampa il seme casuale impostato
  std::cout <<"random seed used: " <<Global::getRandSeed() <<std::endl;

  // Costruisce la rete neurale (passandogli il numero di input, di strati e il
  // numero di unita` per ogni strato)
  units.push_back(outputs);
  nn = new NeuralNetwork(inputs, hlayers+1, units);

  // Costruisce l'algoritmo di back-propagation con i parametri passati
  bp = new BackPropagation();
  bp->setLearningRate(eta);
  bp->setMomentumRate(alpha);
  bp->setRegularizationRate(lambda);

  // Stampa le caratteristiche della rete neurale e dell'algoritmo
  std::cout <<std::endl;
  printNeuralNetworkInfo();
  std::cout <<std::endl;
  printBackPropagationInfo();
  std::cout <<std::endl;

  // Costruisce il trainer con i parametri passati, impostandogli la rete
  // neurale come model e l'algoritmo di back-propagation come algoritmo di
  // training
  tr = new Trainer(nn, bp);
  tr->setDataSet(trfile);
  tr->setFolds(folds);
  tr->setMaxEpochs(maxepochs);
  tr->setShuffleEpochs(shuffle);
  tr->setStopError(stoperr);
  tr->setStopErrorChange(stoperrch, stoperrchep);
  tr->setStopAccuracy(stopacc);
  tr->setThreshold(threshold);

  // Per il numero di partizioni (folds) impostate (attributo maxfolds) esegue
  // il training e (se richiesto) salva i risultati su file.
  for (uint k = 0; k < maxfolds; ++k) {
    tr->resetModel();
    tr->setValidationOn(k);
    if (!trsave.empty()) tr->setSaveResults(trsave+"-"+Global::toString(k+1));
    // avvia il training
    startTimer();
    tr->start();
    stopTimer();
    // aggiorna i risultati
    updateTrainingResults();
    // stampa i risultati ottenuti
    std::cout <<"# training results on fold n. " <<k+1;
    std::cout <<" (of " <<folds <<")" <<std::endl;
    std::cout <<"instances: ";
    if (folds == 1) std::cout <<tr->getDatasetDimension();
    else std::cout <<tr->getDatasetDimension()-tr->getFoldDimension(k);
    std::cout <<" (on dataset of " <<tr->getDatasetDimension() <<")";
    std::cout <<std::endl;
    printTrainingInfo();
    std::cout <<std::endl;
    // salva su file i risultati
    if (!nnsave.empty()) nn->saveOnFile(nnsave+"-"+Global::toString(k+1));
  } // end for k

  // Stampa i risultati medi finali (se e` stato fatto training su piu` folds)
  if (maxfolds > 1) printFinalResults();

  // Elimina le strutture create e termina
  delete nn;
  delete bp;
  delete tr;
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
bool NNTraining::checkParameters ( ) {
  std::vector<std::string> required;
  std::vector<std::string> missingarg;
  std::string strunits;
  // legge i parametri
  // --inputs
  if (Global::getParam("inputs").empty())
    required.push_back("--inputs");
  else if (Global::getParam("inputs") == "inputs")
    missingarg.push_back("--inputs");
  else inputs = Global::toUint(Global::getParam("inputs"));
  // --outputs
  if (Global::getParam("outputs").empty())
    required.push_back("--outputs");
  else if (Global::getParam("outputs") == "outputs")
    missingarg.push_back("--outputs");
  else outputs = Global::toUint(Global::getParam("outputs"));
  // --hlayers
  if (Global::getParam("hlayers").empty())
    required.push_back("--hlayers");
  else if (Global::getParam("hlayers") == "hlayers")
    missingarg.push_back("--hlayers");
  else hlayers = Global::toUint(Global::getParam("hlayers"));
  // --units
  if (Global::getParam("units").empty())
    required.push_back("--units");
  else if (Global::getParam("units") == "units")
    missingarg.push_back("--units");
  else strunits = Global::getParam("units");
  // --eta
  if (Global::getParam("eta").empty())
    required.push_back("--eta");
  else if (Global::getParam("eta") == "eta")
    missingarg.push_back("--eta");
  else eta = Global::toReal(Global::getParam("eta"));
  // --trfile
  if (Global::getParam("trfile").empty())
    required.push_back("--trfile");
  else if (Global::getParam("trfile") == "trfile")
    missingarg.push_back("--trfile");
  else trfile = Global::getParam("trfile");
  // --alpha
  if (Global::getParam("alpha").empty())
    alpha = 0.0; // valore di default
  else if (Global::getParam("alpha") == "alpha")
    missingarg.push_back("--alpha");
  else alpha = Global::toReal(Global::getParam("alpha"));
  // --lambda
  if (Global::getParam("lambda").empty())
    lambda = 0.0; // valore di default
  else if (Global::getParam("lambda") == "lambda")
    missingarg.push_back("--lambda");
  else lambda = Global::toReal(Global::getParam("lambda"));
  // --trsave
  if (Global::getParam("trsave").empty())
    trsave = ""; // valore di default
  else if (Global::getParam("trsave") == "trsave")
    missingarg.push_back("--trsave");
  else trsave = Global::getParam("trsave");
  // --folds
  if (Global::getParam("folds").empty())
    folds = 10; // valore di default
  else if (Global::getParam("folds") == "folds")
    missingarg.push_back("--folds");
  else folds = Global::toUint(Global::getParam("folds"));
  // --maxfolds
  if (Global::getParam("maxfolds").empty())
    maxfolds = folds; // valore di default
  else if (Global::getParam("maxfolds") == "maxfolds")
    missingarg.push_back("--maxfolds");
  else maxfolds = Global::toUint(Global::getParam("maxfolds"));
  // --maxepochs
  if (Global::getParam("maxepochs").empty())
    maxepochs = 0; // valore di default
  else if (Global::getParam("maxepochs") == "maxepochs")
    missingarg.push_back("--maxepochs");
  else maxepochs = Global::toUint(Global::getParam("maxepochs"));
  // --shuffle
  if (Global::getParam("shuffle").empty())
    shuffle = 0; // valore di default
  else if (Global::getParam("shuffle") == "shuffle")
    missingarg.push_back("--shuffle");
  else shuffle = Global::toUint(Global::getParam("shuffle"));
  // --stoperr
  if (Global::getParam("stoperr").empty())
    stoperr = 0.0; // valore di default
  else if (Global::getParam("stoperr") == "stoperr")
    missingarg.push_back("--stoperr");
  else stoperr = Global::toReal(Global::getParam("stoperr"));
  // --stopacc
  if (Global::getParam("stopacc").empty())
    stopacc = 1.1; // valore di default
  else if (Global::getParam("stopacc") == "stopacc")
    missingarg.push_back("--stopacc");
  else stopacc = Global::toReal(Global::getParam("stopacc"));
  // --stoperrch
  if (Global::getParam("stoperrch").empty())
    stoperrch = -1; // valore di default
  else if (Global::getParam("stoperrch") == "stoperrch")
    missingarg.push_back("--stoperrch");
  else stoperrch = Global::toFloat(Global::getParam("stoperrch"));
  // --stoperrchep
  if (Global::getParam("stoperrchep").empty())
    stoperrchep = 10; // valore di default
  else if (Global::getParam("stoperrchep") == "stoperrchep")
    missingarg.push_back("--stoperrchep");
  else stoperrchep = Global::toUint(Global::getParam("stoperrchep"));
  // --threshold
  if (Global::getParam("threshold").empty())
    threshold = 0.5; // valore di default
  else if (Global::getParam("threshold") == "threshold")
    missingarg.push_back("--threshold");
  else threshold = Global::toReal(Global::getParam("threshold"));
  // --nnsave
  if (Global::getParam("nnsave").empty())
    nnsave = ""; // valore di default
  else if (Global::getParam("nnsave") == "nnsave")
    missingarg.push_back("--nnsave");
  else nnsave = Global::getParam("nnsave");
  // controlla se ci sono errori
  if (!required.empty()) {
    std::cout <<"The follow parameters are required (in training mode)";
    std::cout <<std::endl;
    for (std::size_t i = 0; i < required.size(); ++i)
      std::cout <<"  " <<required[i] <<std::endl;
    return false;
  }
  if (!missingarg.empty()) {
    std::cout <<"The follow parameters requires an argument (in training mode)";
    std::cout <<std::endl;
    for (std::size_t i = 0; i < missingarg.size(); ++i)
      std::cout <<"  " <<missingarg[i] <<std::endl;
    return false;
  }
  // controlla i valori dei parametri
  // --units
  std::vector<std::string>* strunitsplit = Global::split(strunits,',');
  if (strunitsplit->size() != hlayers) {
    std::cout <<"Parameter --units has invalid number of values" <<std::endl;
    delete strunitsplit;
    return false;
  }
  for (std::size_t i = 0; i < hlayers; ++i)
    units.push_back( Global::toUint(strunitsplit->at(i)) );
  delete strunitsplit;
  // --folds
  if (folds <= 0) {
    std::cout <<"Parameter --folds must be at least 1" <<std::endl;
    return false;
  }
  // --maxfolds
  if (maxfolds > folds) {
    std::cout <<"Parameter --maxfolds is too large" <<std::endl;
    return false;
  }
  // --stoperr
  if (stoperr < 0) {
    std::cout <<"Parameter --stoperr must be a positive number" <<std::endl;
    return false;
  }
  // --stoperrch
  if (stoperrch < 0) {
    stoperrch = 0;
    stoperrchep = 0;
  }
  // --stopacc
  if (stopacc < 0 || (!Global::getParam("stopacc").empty() && stopacc > 1)) {
    std::cout <<"Parameter --stopacc must a number in [0,1]" <<std::endl;
    return false;
  }
  // --threshold
  if (threshold < 0 || threshold  > 1) {
    std::cout <<"Parameter --threshold must a number in [0,1]" <<std::endl;
    return false;
  }
  return true;
} // End method checkParameters

/**
 * Method printNeuralNetworkInfo
 *
 * Stampa su standard output tutte le informazioni relative alla rete neurale.
 */
void NNTraining::printNeuralNetworkInfo() {
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
 * Method printBackPropagationInfo
 *
 * Stampa su standard output tutte le informazioni relative all'algoritmo di
 * back-propagation.
 */
void NNTraining::printBackPropagationInfo() {
  std::cout <<"# back-propagation algorithm" <<std::endl;
  std::cout <<"learning rate: " <<bp->getLearningRate() <<"\n";
  std::cout <<"momentum rate: " <<bp->getMomentumRate() <<"\n";
  std::cout <<"regularization rate: " <<bp->getRegularizationRate() <<"\n";
  return;
} // End of method printBackPropagationInfo

/**
 * Method updateTrainingResults
 *
 * Aggiorna i valori delle variabili contenenti i risultati del training.
 */
void NNTraining::updateTrainingResults() {
  mtime += getElapsedTime();
  mtcpu += getCpuUsage();
  mepochs += tr->getEpochs();
  mtrerr += tr->getTrainingError();
  mvaerr += tr->getValidationError();
  mtracc += tr->getTrainingAccuracy();
  mvaacc += tr->getValidationAccuracy();
  mtrerrmin += tr->getMinTrainingError().first;
  mvaerrmin += tr->getMinValidationError().first;
  mtraccmax += tr->getMaxTrainingAccuracy().first;
  mvaaccmax += tr->getMaxValidationAccuracy().first;
  return;
} // End of method updateTrainingResults

/**
 * Method printTrainingInfo
 *
 * Stampa su standard output tutte le informazioni relative al training eseguito
 * con un oggetto di tipo Trainer.
 */
void NNTraining::printTrainingInfo() {
  std::cout <<"elapsed time: " <<getElapsedTime() <<" seconds \n";
  std::cout <<"cpu usage: " <<getCpuUsage() <<" seconds \n";
  std::cout <<"epochs: " <<tr->getEpochs() <<"\n";
  std::cout <<"training error: " <<tr->getTrainingError() <<"\n";
  std::cout <<"validation error: " <<tr->getValidationError() <<"\n";
  std::cout <<"training accuracy: " <<tr->getTrainingAccuracy() <<"\n";
  std::cout <<"validation accuracy: " <<tr->getValidationAccuracy() <<"\n";
  std::cout <<"tr. error min.: " <<tr->getMinTrainingError().first
      <<" (" <<tr->getMinTrainingError().second  <<")\n";
  std::cout <<"va. error min.: " <<tr->getMinValidationError().first
      <<" (" <<tr->getMinValidationError().second  <<")\n";
  std::cout <<"tr. accuracy max.: " <<tr->getMaxTrainingAccuracy().first
      <<" (" <<tr->getMaxTrainingAccuracy().second  <<")\n";
  std::cout <<"va. accuracy max.: " <<tr->getMaxValidationAccuracy().first
      <<" (" <<tr->getMaxValidationAccuracy().second  <<")\n";
  return;
} // End of method printTrainingInfo

/**
 * Method printFinalResults
 *
 * Stampa su standard output tutte le informazioni finali del training.
 */
void NNTraining::printFinalResults() {
  std::cout <<"# final training results (on " <<maxfolds <<" folds)\n";
  std::cout <<"time (avg): " <<mtime/maxfolds <<std::endl;
  std::cout <<"cpu usage (avg): " <<mtcpu/maxfolds <<std::endl;
  std::cout <<"epochs (avg): " <<float(mepochs)/maxfolds <<std::endl;
  std::cout <<"tr. error (avg): " <<mtrerr/maxfolds <<std::endl;
  std::cout <<"va. error (avg): " <<mvaerr/maxfolds <<std::endl;
  std::cout <<"tr. accuracy (avg): " <<mtracc/maxfolds <<std::endl;
  std::cout <<"va. accuracy (avg): " <<mvaacc/maxfolds <<std::endl;
  std::cout <<"min. tr. err. (avg): " <<mtrerrmin/maxfolds <<std::endl;
  std::cout <<"min. va. err. (avg): " <<mvaerrmin/maxfolds <<std::endl;
  std::cout <<"max. tr. acc. (avg): " <<mtraccmax/maxfolds <<std::endl;
  std::cout <<"max. va. acc. (avg): " <<mvaaccmax/maxfolds <<std::endl;
  return;
} // End of method printFinalResults

/**
 * Method startTimer
 *
 * Memorizza nelle variabili time_start e tcpu_start il tempo e il numero di
 * clock della cpu al momento dell'invocazione del metodo.
 * Attraverso i metodi getElapsedTime e getCpuUsage si ottengono rispettivamente
 * il tempo (in secondi) e il tempo di utilizzo della cpu (in secondi) trascorsi
 * dall'invocazione di questo metodo all'invocazione del metodo stopTimer.
 */
void NNTraining::startTimer() {
  gettimeofday(&time_start, NULL);
  tcpu_start = clock();
  return;
} // End method startTimer

/**
 * Method stopTimer
 *
 * Memorizza nelle variabili time_stop e tcpu_stop il tempo e il numero di clock
 * della cpu al momento dell'invocazione del metodo.
 * Attraverso i metodi getElapsedTime e getCpuUsage si ottengono rispettivamente
 * il tempo (in secondi) e il tempo di utilizzo della cpu (in secondi) trascorsi
 * dall'invocazione del metodo startTimer all'invocazione di questo metodo.
 */
void NNTraining::stopTimer() {
  gettimeofday(&time_end, NULL);
  tcpu_end = clock();
  return;
} // End method stopTimer

/**
 * Method getElapsedTime
 *
 * Restituisce il numero di secondi trascorsi dall'invocazione del metodo
 * startTimer all'invocazione del metodo stopTimer.
 */
double NNTraining::getElapsedTime() {
  double s1 = time_start.tv_sec+(time_start.tv_usec/1000000.0);
  double s2 = time_end.tv_sec+(time_end.tv_usec/1000000.0);
  return s2-s1;
} // End method getElapsedTime

/**
 * Method getCpuUsage
 *
 * Restituisce i secondi di utilizzo della cpu dall'invocazione del metodo
 * startTimer all'invocazione del metodo stopTimer.
 */
double NNTraining::getCpuUsage ( ) {
  return (tcpu_end-tcpu_start) / double(CLOCKS_PER_SEC);
} // End method getCpuUsage
