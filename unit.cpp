#include "unit.h"

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include "global.h"
#include "exception.h"

typedef Global::uint uint;
typedef Global::real real;

/**
 * Constructor Unit()
 *
 * Costruisce un'unita con il numero di inputs passato come parametro ed i
 * relativi pesi inizializzati in modo casuale nell'intervallo [-0.7,+0.7].
 */
Unit::Unit(uint numberOfInputs) :
  numberOfInputs(numberOfInputs),
  numberOfWeights(numberOfInputs+1),
  inputs(numberOfInputs+1,0),
  weights(numberOfInputs+1,0),
  lastOutput(0)
{
  inputs[0] = 1;
  initWeightsRandom();
  return;
} // End constructor Unit()

/**
 * Distructor ~Unit()
 */
Unit::~Unit() { }

/**
 * Copy constructor
 *
 * Costruisce una unita` con un numero di input (e quindi anche un numero di
 * pesi) uguale all'unita` passata, ma con inputs inizializzati a 0 e pesi
 * casuali in [-0.7, +0.7].
 */
Unit::Unit(const Unit& unit) :
  numberOfInputs(unit.numberOfInputs),
  numberOfWeights(unit.numberOfWeights),
  inputs(numberOfInputs+1,0),
  weights(numberOfWeights,0),
  lastOutput(0)
{
  inputs[0] = 1;
  initWeightsRandom();
  return;
} // End copy constructor

// ==============
// PUBLIC METHODS
// ==============

/**
 * Method setInput
 *
 * Imposta il valore nell'i-esimo input con il valore passato. L'indice dell'
 * input (parametro i) deve essere nell'intervallo [1, numberOfInputs].
 */
void Unit::setInput(uint i, real input) {
  if (i < 1 || i > numberOfInputs)
    throw std::out_of_range("In class Unit");
  inputs[i] = input;
  return;
} // End method setInput()

/**
 * Method setInputs
 *
 * Imposta gli inputs dell'unita` con i valori passati. La dimensione del
 * vettore "inputs" deve essere "numberOfInputs"
 */
void Unit::setInputs(const std::vector<real>& inputs) {
  if (inputs.size() != numberOfInputs)
    return;
  for (uint i = 0; i < numberOfInputs; ++i)
    this->inputs[i+1] = inputs[i];
  return;
} // End method setInputs

/**
 * Method setWeight
 *
 * Imposta il valore nell'i-esimo peso dell'unita` con il valore passato.
 * L'indice dei pesi (parametro i) deve essere nell'intervallo
 * [0, numberOfWeights-1]
 */
void Unit::setWeight(uint i, real weight) {
  if (i >= numberOfWeights)
    throw std::out_of_range("In class Unit");
  weights[i] = weight;
  return;
} // End method setWeight()

/**
 * Method setWeights
 *
 * Imposta i pesi dell'unita` con i valori passati.
 */
void Unit::setWeights(const std::vector<real>& weights) {
  if (weights.size() != this->weights.size())
    return;
  this->weights = weights;
  return;
} // End method setWeights

/**
 * Method sumToWeight
 *
 * Somma all'i-esimo peso il valore passato
 */
void Unit::sumToWeight(uint i, real value) {
  if (i >= numberOfWeights)
    throw std::out_of_range("In class Unit");
  weights[i] += value;
  return;
} // End method sumToWeight

/**
 * Method getLastInput
 *
 * Restituisce il valore dell'ultimo input inserito in posizione i. L'indice i
 * puo` essere nell'intervallo [0, numberOfInputs], dove in posizione 0 c'e`
 * l'input con valore sempre 1.
 */
real Unit::getLastInput(uint i) const {
  if (i > numberOfInputs)
    throw std::out_of_range("In class Unit");
  return inputs[i];
} // End method getLastInput

/**
 * Method getLastOutput
 *
 * Restituisce l'ultimo output calcolato (senza ricalcolarlo).
 */
real Unit::getLastOutput() const {
  return lastOutput;
} // End method getLastOutput

/**
 * Method getNumberOfInputs
 *
 * Restituisce il numero di inputs dell'unita` (uguale al numero di pesi meno 1
 * perche` nei pesi c'e` w0)
 */
uint Unit::getNumberOfInputs() const {
  return numberOfInputs;
} // End method getNumberOfInputs

/**
 * Method getNumberOfWeights
 *
 * Restituisce il numero di pesi dell'unita` (uguale al numero di inputs piu` 1
 * perche` c'e` il peso w0)
 */
uint Unit::getNumberOfWeights() const {
  return numberOfWeights;
} // End method getNumberOfWeights

/**
 * Method getWeight
 *
 * Restituisce il valore dell'i-esimo peso. L'indice dei pesi (parametro i) deve
 * essere nell'intervallo [0, numberOfWeights-1]
 */
real Unit::getWeight(uint i) const {
  if (i >= numberOfWeights)
    throw std::out_of_range("In class Unit");
  return weights[i];
} // End method getWeight

/**
 * Method computeOutput
 *
 * Calcola e restituisce l'output a partire dagli ultimi inputs impostati
 */
real Unit::computeOutput() {
  return lastOutput = calcOutput();
} // End method computeOutput

/**
 * Method write
 *
 * Stampa l'unita` sullo stream di output (passato come parametro). L'unita`
 * viene scritta nel seguente formato:
 *   n,weight(1),...,weight(n)
 */
const Unit& Unit::write(std::ostream& os) const {
  os <<numberOfWeights;
  // modifica la precisione della stampa di numeri floating point
  std::streamsize prprec = os.precision(20);
  std::ios::fmtflags prflag = os.setf(std::ios::scientific,
      std::ios::floatfield);
  // scrive l'unita`
  for (uint i = 0; i < numberOfWeights; ++i)
    os <<',' <<weights[i];
  // ripristina i valori della precisione
  os.precision(prprec);
  os.setf(prflag, std::ios::floatfield);
  return *this;
} // End method write

/**
 * Method read
 *
 * Legge un'unita` dallo stream passato. L'unita` viene letta nel seguente
 * formato:
 *   n,weight(1),...,weight(n) \n
 */
Unit& Unit::read(std::istream& is) {
  // legge dallo stream i valori
  std::string line;
  std::getline(is, line);
  std::vector<std::string>* w = Global::split(Global::trim(line),',');
  if ( !(w->size() >= 2) )
    throw read_error("In Unit::read");
  if ( !((w->size()-1) == Global::toUint(w->at(0))) )
    throw read_error("In Unit::read");
  // costruisce la nuova unita` con i valori letti
  numberOfWeights = Global::toUint(w->at(0));
  weights.clear();
  numberOfInputs = numberOfWeights-1;
  inputs.clear();
  weights.push_back(Global::toReal(w->at(1)));
  inputs.push_back(1);
  for (uint i = 2; i <= numberOfWeights; ++i) {
    weights.push_back(Global::toReal(w->at(i)));
    inputs.push_back(0.0);
  }
  lastOutput = 0.0;
  delete w;
  return *this;
} // End method read

// ===============
// PRIVATE METHODS
// ===============

/**
 * Method initWeightsRandom
 *
 * Inizializza in modo casuale il valore dei pesi dell'unita`
 */
void Unit::initWeightsRandom() {
  std::for_each(weights.begin(), weights.end(), setRandomValue);
  return;
} // End method initWeightsRandom

/**
 * Method setRandomValue
 *
 * Assegna un numero random nell'intervallo [-0.7,+0.7] (escluso lo 0) alla
 * variabile passata
 */
void Unit::setRandomValue(real& val) {
  do {
    val = ( (Global::getRand(0,1400)-700) / 1000.0 );
  } while (val == 0);
  return;
} // End method setRandomValue

/**
 * Method calcOutput
 *
 * Calcola l'output dell'unita`
 */
real Unit::calcOutput() const {
  real net = 0.0;
  for (uint i = 0; i < numberOfWeights; ++i)
    net += weights[i]*inputs[i];
  return activationFunction(net);
} // end method calcOutput

/**
 * Method activationFunction()
 *
 * Calcola la funzione di attivazione f(net) = 1/(1 + e^(-net))
 */
real Unit::activationFunction(real net) const {
  return 1 / ( 1 + exp(-net*1) );
} // end method activation

// =================
// RELATED FUNCTIONS
// =================

/**
 * Function operator<<
 *
 * Stampa l'unita` sullo stream out (passato come parametro). Per il formato
 * con cui viene scritta vedere il metodo write
 */
std::ostream& operator<<(std::ostream& os, const Unit& u) {
  u.write(os);
  return os;
} // End function operator<<

/**
 * Function operator>>
 *
 * Legge una unita` dallo stream passato. Per il formato con cui viene letta
 * l'unita` vedere il metodo read
 */
std::istream& operator>>(std::istream& is, Unit& u) {
  u.read(is);
  return is;
} // End function operator<<

