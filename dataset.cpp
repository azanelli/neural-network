#include "dataset.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include "global.h"
#include "exception.h"

typedef Global::uint uint;
typedef Global::real real;

/**
 * Default constructor
 *
 * Costruisce un dataset vuoto
 */
Dataset::Dataset() :
    folds(0), vafold(0)
{ } // End default constructor

// ==============
// PUBLIC METHODS
// ==============

/**
 * Method loadDataSet
 *
 * Carica da file (in formato csv) il dataset di istanze. Le istanze nel file
 * devono essere una per riga e nella forma:
 *    id, x1, ..., xn, y1, ..., ym
 * dove id e` un identificatore univoco della istanza, x sono gli inputs, y sono
 * gli outputs. La funzione prende i seguenti parametri:
 *   - file : nome del file (.csv) da cui caricare il dataset
 *   - ninputs : numero degli inputs per ogni istanza
 *   - noutputs : numero degli outputs per ogni istanza
 */
void Dataset::load(const std::string& filename, uint ninputs, uint noutputs) {
  std::string line;
  std::ifstream file(filename.c_str());
  if (!file.is_open()) throw file_error("In Dataset::load");
  dataset.clear();
  av.clear();
  while (file.good()) {
    std::getline(file, line);
    Global::trim(line, " \t");
    if (!line.empty() && line[0] != '#') {
      std::vector<std::string>* csvline = Global::split(line, ',');
      assert( ninputs + noutputs + 1 == csvline->size() );
      dataset.push_back(Instance());
      dataset.back().id = Global::trim(csvline->at(0));
      for (uint i = 0; i < ninputs; ++i)
        dataset.back().input.push_back(Global::toReal(csvline->at(i+1)));
      for (uint i = 0; i < noutputs; ++i)
        dataset.back().output.push_back(Global::toReal(
            csvline->at(i+1+ninputs) ));
      delete csvline;
    } // end if
  } // end while (file.good())
  av.resize(dataset.size());
  for (uint i = 0; i < av.size(); ++i) av[i] = i;
  return;
} // End method load

/**
 * Method setFolds
 *
 * Divide il dataset in n partizioni uguali. Se n e` 0 viene fatto il merge
 * (vedere il metodo merge), se n e` 1 l'unico folds e` l'intero dataset che
 * costituisce il training set (non c'e` validation set).
 * Il numero di folds dev'essere minore della dimensione del dataset.
 */
void Dataset::setFolds(uint n) {
  if (n > dataset.size()) throw std::out_of_range("In Dataset::setFolds");
  if (n == 0) {
    merge();
    return;
  }
  this->folds = n;
  this->vafold = 0;
  makeTrAccessVector();
  return;
} // End method setFolds

/**
 * Method setValidationFold
 *
 * Imposta il fold per il validation set. L'indice k dev'essere compreso tra 0
 * e n-1, con n il numero di folds. Se il numero di folds e` 1 la validazione
 * non puo` essere fatta (il validation set e` vuoto).
 * L'invocazione di questo metodo porta anche al ripristino dell'ordine delle
 * istanze nelle partizioni, percui per un ordine casuale sul training set
 * si deve ri-invocare il metodo randomShuffleTrainingSet.
 */
void Dataset::setValidationFold (uint k) {
  if (k >= folds) throw std::out_of_range("In Dataset::setValidationFold");
  if (folds == 1) return;
  this->vafold = k;
  makeTrAccessVector();
} // End method setValidationFold

/**
 * Method merge
 *
 * Elimina le partizioni del dataset
 */
void Dataset::merge ( ) {
  folds = 0;
  vafold = 0;
  trav.clear();
} // End method merge

/**
 * Method isEmpty
 *
 * Restituisce true se il dataset e` vuoto (non contiene istanze), false
 * altrimenti
 */
bool Dataset::isEmpty ( ) const {
  return dataset.empty();
} // End method isEmpty

/**
 * Method getSize
 *
 * Restituisce la dimensione del dataset (numero di istanze presenti)
 */
uint Dataset::getSize() const {
  return dataset.size();
} // End method getSize

/**
 * Method getFolds
 *
 * Restituisce il numero di folds impostati
 */
uint Dataset::getFolds() const {
  return folds;
} // End method getFolds

/**
 * Method getFoldSize
 *
 * Restituisce la dimensione dell'i-esimo fold
 */
uint Dataset::getFoldSize(uint i) const {
  if (i >= folds) throw std::out_of_range("In Dataset::getFoldSize");
  return foldDimension(i);
} // End method getFoldSize

/**
 * Method getTrSetSize
 *
 * Restituisce la dimensione (numero di istanze) del training set
 */
uint Dataset::getTrSetSize() const {
  return trav.size();
} // End method getTrSetSize

/**
 * Method getVaSetSize
 *
 * Restituisce la dimensione (numero di istanze) del validation set impostato
 */
uint Dataset::getVaSetSize() const {
  if (folds <= 1)
    return 0;
  return foldDimension(vafold);
} // End method getVaSetSize

/**
 * Method getId
 *
 * Restituisce un riferimento (costante) all'id dell'i-esimo elemento
 */
const std::string& Dataset::getId (uint i) const {
  if (i >= dataset.size()) throw std::out_of_range("In Dataset::operator[]");
  return this->at(i).id;
} // End method getId

/**
 * Method getInputs
 *
 * Restituisce un riferimento (costante) all'input dell'i-esimo elemento
 */
const std::vector<real>& Dataset::getInputs ( uint i ) const {
  if (i >= dataset.size()) throw std::out_of_range("In Dataset::operator[]");
  return this->at(i).input;
} // End method getInputs

/**
 * Method getOutputs
 *
 * Restituisce un riferimento (costante) output dell'i-esimo elemento
 */
const std::vector<real>& Dataset::getOutputs ( uint i ) const {
  if (i >= dataset.size()) throw std::out_of_range("In Dataset::operator[]");
  return this->at(i).output;
} // End method getOutputs

/**
 * Method trAt
 *
 * Restituisce l'i-esimo elemento (istanza) del training set
 */
const Dataset::Instance& Dataset::trAt(uint i) {
  if (i >= trav.size()) throw std::out_of_range("In Dataset::trAt");
  return this->at(trav[i]);
} // End method trAt

/**
 * Method vaAt
 *
 * Restituisce l'i-esimo elemento (istanza) del validation set
 */
const Dataset::Instance& Dataset::vaAt(uint i) {
  if (i >= foldDimension(vafold) || folds <= 1)
    throw std::out_of_range("In Dataset::vaAt");
  return this->at(startIndexFold(vafold)+i);
} // End method vaAt

/**
 * Method at
 *
 * Restituisce un riferimento (costante) all'i-esimo elemento (istanza) del
 * dataset
 */
const Dataset::Instance& Dataset::at(uint i) const {
  if (i >= dataset.size()) throw std::out_of_range("In Dataset::operator[]");
  return dataset[av[i]];
} // End method at

/**
 * Method operator[]
 *
 * Restituisce un riferimento (costante) all'i-esimo elemento (istanza) del
 * dataset
 */
const Dataset::Instance& Dataset::operator[](uint i) const {
  if (i >= dataset.size()) throw std::out_of_range("In Dataset::operator[]");
  return this->at(i);
} // End method operator[]

/**
 * randomShuffleTrainingSet
 *
 * Crea una permutazione casuale delle istanze del training set
 */
void Dataset::randomShuffleTrainingSet() {
  for (uint i = trav.size(); i != 0; --i)
    std::swap( trav[i-1], trav[Global::getRand(0,i-1)] );
  return;
} // End method randomShuffleTrainingSet

/**
 * randomShuffle
 *
 * Crea una permutazione casuale del dataset. Se sono impostate dei folds
 * i loro elementi vengono mischiati in modo casuale
 */
void Dataset::randomShuffle() {
  for (uint i = av.size(); i != 0; --i)
    std::swap( av[i-1], av[Global::getRand(0,i-1)] );
  return;
} // End method randomShuffle

/**
 * Method restore
 *
 * Ripristina il dataset, eliminando le partizioni create e riportandolo all'
 * ordinamento iniziale
 */
void Dataset::restore() {
  merge();
  for (uint i = 0; i < av.size(); ++i) av[i] = i;
  return;
} // End method restore

// ===============
// PRIVATE METHODS
// ===============

/**
 * Method makeTrAccessVector
 *
 * Costruisce il vettore di accesso al training set, leggendo la variabile
 * vafold per conoscere la partizione di validation. Se il numero di folds e`
 * 1 allora viene costruito sull'intero dataset
 */
inline
void Dataset::makeTrAccessVector() {
  trav.clear();
  if (folds == 1)
    for (uint i = 0; i < dataset.size(); ++i) trav.push_back(i);
  else
    for (uint i = 0; i < dataset.size(); ++i)
      if (i < startIndexFold(vafold) || i >= endIndexFold(vafold))
        trav.push_back(i);
  return;
} // End method makeTrAccessVector

/**
 * Method startIndexFold
 *
 * Restituisce l'indice del primo elemento della k-esima partizione
 */
inline
uint Dataset::startIndexFold(uint k) const {
  assert(k < folds);
  uint rest = dataset.size()%folds;
  if (k <= rest) return k * ( floor(dataset.size()/double(folds)) + 1 );
  return (k * floor(dataset.size()/double(folds)) ) + rest;
} // End method startIndexFold

/**
 * Method endIndexFold
 *
 * Restituisce l'indice dell'ultimo elemento della k-esima partizione
 */
inline
uint Dataset::endIndexFold(uint k) const {
  assert(k < folds);
  if (k == (folds-1)) return dataset.size();
  return startIndexFold(k+1);
} // End method endIndexFold

/**
 * Method foldDimension
 *
 * Restituisce la dimensione della k-esima partizione
 */
inline
uint Dataset::foldDimension(uint k) const {
  assert(k < folds);
  return endIndexFold(k) - startIndexFold(k);
} // End method foldDimension
