#include "backpropagation.h"

#include <cassert>
#include "global.h"
#include "neuralnetwork.h"

typedef Global::uint uint;
typedef Global::real real;

/**
 * Constructor BackPropagation
 *
 * Costruisce un oggetto di tipo BackPropagation con valori di default.
 * Per applicare un passo dell'algoritmo con il metodo compute e` necessario
 * prima impostare un modello (una rete neurale) con il metodo setModel.
 */
BackPropagation::BackPropagation() :
    neuralnetwork(NULL),
    eta(0.0),
    lambda(0.0),
    alfa(0.0),
    momentumtable(NULL)
{ } // End constructor BackPropagation

/**
 * Destructor ~BackPropagation
 */
BackPropagation::~BackPropagation() {
  deleteMomentumTable();
  return;
} // End constructor ~BackPropagation

// ==============
// PUBLIC METHODS
// ==============

/**
 * Method setModel
 *
 * Imposta la rete neurale sulla quale applicare l'algoritmo di
 * back-propagation
 */
void BackPropagation::setModel(NeuralNetwork* neuralnetwork) {
  this->neuralnetwork = neuralnetwork;
  makeMomentumTable();
  return;
} // End method setModel

/**
 * Method setLearningRate
 *
 * Imposta il learning rate (eta) dell'algoritmo
 */
void BackPropagation::setLearningRate(real eta) {
  this->eta = eta;
} // End method setLearningRate

/**
 * Method setMomentumRate
 *
 * Imposta il rate del momentum (alfa)
 */
void BackPropagation::setMomentumRate(real alfa) {
  this->alfa = alfa;
} // End method setMomentumRate

/**
 * Method setRegularizationRate
 *
 * Imposta il rate per la regolarizzazione (lambda)
 */
void BackPropagation::setRegularizationRate(real lambda) {
  this->lambda = lambda;
} // End method setRegularizationRate

/**
 * Method getLearningRate
 *
 * Restituisce il learning rate (eta) utilizzato
 */
real BackPropagation::getLearningRate() const {
  return eta;
} // End method getLearningRate

/**
 * Method getMomentumRate
 *
 * Restituisce il rate del momentum (alfa) utilizzato
 */
real BackPropagation::getMomentumRate() const {
  return alfa;
} // End method getMomentumRate

/**
 * Method getRegularizationRate
 *
 * Restituisce il rate per la regolarizzazione (lambda) utilizzato
 */
real BackPropagation::getRegularizationRate() const {
  return lambda;
} // End method getRegularizationRate

/**
 * Method compute
 *
 * Applica un passo dell'algoritmo back-propagation alla rete neurale.
 * I parametri passati sono i seguenti:
 *   - inputs : vettore con gli inputs dell'istanza di training
 *   - desiredResponse : risposta desiderata per gli inputs passati
 */
void BackPropagation::compute(const std::vector<real>& inputs,
    const std::vector<real>& desiredResponse) {
  assert(inputs.size() == neuralnetwork->getInputs().size());
  assert(desiredResponse.size() == neuralnetwork->getNumberOfOutputs());
  // Forward phase
  neuralnetwork->setInputs(inputs);
  neuralnetwork->compute();
  // Backward phase
  const uint nLayers = neuralnetwork->getNumberOfLayers();
  assert(nLayers >= 2);
  real delta = 0;
  real unitOutput = 0;
  // Calcolo sullo strato di output
  uint curLayer = nLayers - 1;
  std::vector<real> errorVector (
      neuralnetwork->getLayerDimension(curLayer-1), 0 );
  for (uint i = 0; i < neuralnetwork->getLayerDimension(curLayer); ++i) {
    // calcolo del gradiente locale
    unitOutput = neuralnetwork->getOutput(i);
    delta = localGradient(desiredResponse[i]-unitOutput, unitOutput);
    // propagazione dell'errore e aggiornamento dei pesi
    assert( errorVector.size() ==
        neuralnetwork->getNumberOfWeight(curLayer,i)-1 );
    // aggiornamento di w0 (senza regolarizzazione, lambda = 0)
    updateWeight(curLayer, i, 0, eta, delta, 0, alfa);
    for (uint j = 0; j < errorVector.size(); ++j) {
      // propagazione dell'errore
      errorVector[j] += delta * neuralnetwork->getWeight(curLayer,i,j+1);
      // aggiornamento dei pesi
      updateWeight(curLayer, i, j+1, eta, delta, lambda, alfa);
    } // end for j
  } // end for i
  // Calcolo sugli strati nascosti
  std::vector<real> deltaVector;
  curLayer = nLayers - 2;
  for (uint t = 0; t <= nLayers-2; ++t, --curLayer) {
    // calcolo del gradiente locale
    deltaVector.clear();
    deltaVector.resize(neuralnetwork->getLayerDimension(curLayer), 0);
    assert( errorVector.size() == deltaVector.size() );
    for (uint i = 0; i < deltaVector.size(); ++i) {
      unitOutput = neuralnetwork->getUnitOutput(curLayer, i);
      deltaVector[i] = localGradient(errorVector[i], unitOutput);
    } // end for i
    // propagazione dell'errore e aggiornamento dei pesi
    errorVector.clear();
    if (curLayer > 0)
      errorVector.resize(neuralnetwork->getLayerDimension(curLayer-1), 0);
    else
      errorVector.resize(neuralnetwork->getInputs().size(), 0);
    for (uint i = 0; i < deltaVector.size(); ++i) {
      assert( errorVector.size() ==
          neuralnetwork->getNumberOfWeight(curLayer,i)-1 );
      // aggiornamento w0 (senza regolarizzazione, lambda = 0)
      updateWeight(curLayer, i, 0, eta, deltaVector[i], 0, alfa);
      for (uint j = 0; j < errorVector.size(); ++j) {
        // propagazione errore
        errorVector[j] +=
            deltaVector[i] * neuralnetwork->getWeight(curLayer,i,j+1);
        // aggiornamento dei pesi
        updateWeight(curLayer, i, j+1, eta, deltaVector[i], lambda, alfa);
      } // end for j
    } // end for i
  } // end for curLayer
  return;
} // End method compute

// ===============
// PRIVATE METHODS
// ===============

/**
 * Method localGradient
 *
 * Calcola il gradiente locale di una unita` passandogli i seguenti parametri:
 *   - error : errore dell'unita`
 *   - output : output dell'unita`
 */
inline
real BackPropagation::localGradient(real error, real output) const {
  return error * (1 * output * (1 - output) );
} // End method localGradient

/**
 * Method updateWeight
 *
 * Aggiorna il peso dell'unita` specificata e mantiene aggiornata la
 * momentumtable (che contiene tutte le precedenti modifiche ai pesi).
 */
inline
void BackPropagation::updateWeight(uint layer, uint unit, uint weight, real eta,
    real delta, real lambda, real alfa) const {
  // Calcolo del valore da aggiungere al peso
  real deltaweight =
      eta * delta * neuralnetwork->getUnitInput(layer,unit,weight) -
      2 * eta * lambda * neuralnetwork->getWeight(layer,unit,weight) +
      alfa * momentumtable->at(layer)->at(unit);
  // Aggiorna il peso
  neuralnetwork->sumToWeight(layer, unit, weight, deltaweight);
  // Aggiorna la tabella del momentum
  momentumtable->at(layer)->at(unit) = deltaweight;
  return;
} // End method updateWeight

/**
 * Method makeMomentumTable
 *
 * Costruisce la tabella per mantenere i valori precedenti delle modifiche ai
 * pesi, utilizzata per il calcolo del momentum.
 */
void BackPropagation::makeMomentumTable() {
  if (neuralnetwork == NULL)
    return;
  if (momentumtable != NULL)
    deleteMomentumTable();
  momentumtable = new std::vector< std::vector<real>* > (
      neuralnetwork->getNumberOfLayers(), NULL );
  for (uint i = 0; i < momentumtable->size(); ++i)
    momentumtable->at(i) = new std::vector<real>(
        neuralnetwork->getLayerDimension(i), 0.0 );
  return;
} // End method makeMomentumTable

/**
 * Method deleteMomentumTable
 *
 * Elimina la momentum table.
 */
void BackPropagation::deleteMomentumTable() {
  if (momentumtable == NULL)
    return;
  for (uint i = 0; i < momentumtable->size(); ++i)
    delete momentumtable->at(i);
  delete momentumtable;
  momentumtable = NULL;
  return;
} // End method deleteMomentumTable

