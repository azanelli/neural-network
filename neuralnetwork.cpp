#include "neuralnetwork.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdexcept>
#include "global.h"
#include "exception.h"
#include "unit.h"

typedef Global::uint uint;
typedef Global::real real;

/**
 * Constructor NeuralNetwork
 *
 * Crea una rete neurale "vuota" (con zero inputs, e zero strati) che di per se
 * non ha senso, ma si deve utilizzare per leggerla da input attraverso
 * l'operatore >> o il metodo read.
 * Non e` garantito che altri metodi al di fuori della lettura funzionino
 * correttamente su una rete neurale vuota.
 */
NeuralNetwork::NeuralNetwork() :
  ninputs(0),
  nlayers(0),
  inputs(ninputs, 0.0),
  network(new std::vector< std::vector<Unit>* >(nlayers, NULL)),
  lastOutput(0, 0.0)
{ } // End constructor NeuralNetwork

/**
 * Constructor NeuralNetwork
 *
 * Costruisce una rete neurale con i seguenti parametri:
 *   - ninputs : numero di inputs
 *   - nlayers : numero di strati compreso quello di output
 *   - nunits  : array (di dimensione nlayers) con la dimensione di ogni strato
 */
NeuralNetwork::NeuralNetwork(uint ninputs, uint nlayers,
    const std::vector<uint>& nunits) :
  ninputs(ninputs),
  nlayers(nlayers),
  inputs(ninputs, 0),
  network(new std::vector< std::vector<Unit>* >(nlayers, NULL)),
  lastOutput(nunits[nlayers-1], 0)
{
  // Costruisce la rete di unita`
  uint dimPrevLayer = ninputs;
  for (uint i = 0; i < nlayers; ++i) {
    network->at(i) = new std::vector<Unit>( nunits[i], Unit(dimPrevLayer) );
    dimPrevLayer = nunits[i];
  }
  return;
} // End constructor NeuralNetwork

/**
 * Copy constructor NeuralNetwork
 *
 * Costruisce una rete neurale identica a quella passata come parametro
 */
NeuralNetwork::NeuralNetwork ( const NeuralNetwork& neuralnetwork ) :
  ninputs(neuralnetwork.ninputs),
  nlayers(neuralnetwork.nlayers),
  inputs(neuralnetwork.inputs),
  network(new std::vector< std::vector<Unit>* >(nlayers, NULL)),
  lastOutput(neuralnetwork.lastOutput)
{
  // copia la rete di unita`
  for (uint i = 0; i < network->size(); ++i) {
    network->at(i) =
        new std::vector<Unit>(neuralnetwork.network->at(i)->size());
    for (uint j = 0; j < network->at(i)->size(); ++j)
      network->at(i)->at(j) = neuralnetwork.network->at(i)->at(j);
  } // end for i
  return;
} // End of copy constructor

/**
 * Destructor ~NeuralNetwork
 */
NeuralNetwork::~NeuralNetwork() {
  for (uint i = 0; i < nlayers; ++i)
    delete network->at(i);
  delete network;
  return;
} // End destructor ~NeuralNetwork

// ==============
// PUBLIC METHODS
// ==============

/**
 * Metod setInput
 *
 * Imposta l'i-esimo input con il valore passato
 */
void NeuralNetwork::setInput(uint i, real input) {
  if (i >= ninputs)
    throw std::out_of_range("In NeuralNetwork::setInput");
  inputs[i] = input;
  return;
} // End method setInput

/**
 * Metod setInputs
 *
 * Imposta gli inputs con i valori passati
 */
void NeuralNetwork::setInputs(const std::vector<real>& inputs) {
  if (inputs.size() != ninputs)
    return;
  this->inputs = inputs;
  return;
} // End method setInputs

/**
 * Metod setWeight
 *
 * Imposta un peso di una unita` identificata dai seguenti parametri:
 *   - layer  : indice dello strato
 *   - unit   : indice dell'unita` nello strato scelto
 *   - index  : indice del peso nell'unita` scelta
 *   - weight : nuovo peso
 */
void NeuralNetwork::setWeight(uint layer, uint unit, uint index, real weight) {
  if (layer >= nlayers || unit >= network->at(layer)->size() ||
      index >= network->at(layer)->at(unit).getNumberOfWeights() )
    throw std::out_of_range("In NeuralNetwork::setWeight");
  network->at(layer)->at(unit).setWeight(index, weight);
  return;
} // End method setWeight

/**
 * Metod getInput
 *
 * Restituisce l'i-esimo input della rete neurale
 */
real NeuralNetwork::getInput(uint i) const {
  if (i >= ninputs)
    throw std::out_of_range("In NeuralNetwork::getInput");
  return inputs[i];
} // End method getInput

/**
 * Metod getInputs
 *
 * Restituisce un riferimento ad un vettore contenente gli ultimi inputs
 * impostati nella rete.
 */
const std::vector<real>& NeuralNetwork::getInputs() const {
  return inputs;
} // End method getInputs

/**
 * Method getOutputs
 *
 * Restituisce un riferimento ad un vettore contenente gli ultimi outputs
 * calcolati (con il metodo compute).
 */
const std::vector<real>& NeuralNetwork::getOutputs() const {
  return lastOutput;
} // End method getOutput

/**
 * Method getOutput
 *
 * Restituisce l'i-esimo ultimo output calcolato (con il metodo compute)
 */
real NeuralNetwork::getOutput(uint i) const {
  if (i > lastOutput.size())
    throw std::out_of_range("In NeuralNetwork::getOutput");
  return lastOutput[i];
} // End method getOutput

/**
 * Metod getUnitInput
 *
 * Restituisce l'input dell'unita` identificata dai parametri:
 *   - layer  : indice dello strato
 *   - unit   : indice dell'unita` nello strato scelto
 *   - index  : indice dell'input nell'unita` scelta
 */
real NeuralNetwork::getUnitInput(uint layer, uint unit, uint index ) const {
  if (layer >= nlayers || unit >= network->at(layer)->size() ||
        index >= network->at(layer)->at(unit).getNumberOfWeights() )
    throw std::out_of_range("In NeuralNetwork::getUnitInput");
  return network->at(layer)->at(unit).getLastInput(index);
} // End method getUnitInput

/**
 * Method getUnitOutput
 *
 * Restituisce l'ultimo output (calcolato con il metodo compute) dell'unita`
 * specificata secondo i seguenti parametri:
 *   - layer  : indice dello strato
 *   - unit   : indice dell'unita` nello strato scelto
 */
real NeuralNetwork::getUnitOutput(uint layer, uint unit) const {
  if (layer >= nlayers || unit >= network->at(layer)->size() )
    throw std::out_of_range("In NeuralNetwork::getUnitOutput");
  return network->at(layer)->at(unit).getLastOutput();
} // End method getUnitOutput

/**
 * Metod getWeight
 *
 * Restituisce il peso di un'unita` identificata dai seguenti parametri:
 *   - layer  : indice dello strato
 *   - unit   : indice dell'unita` nello strato scelto
 *   - index  : indice del peso nell'unita` scelta
 */
real NeuralNetwork::getWeight(uint layer, uint unit, uint index) const {
  if (layer >= nlayers || unit >= network->at(layer)->size() ||
      index >= network->at(layer)->at(unit).getNumberOfWeights() )
    throw std::out_of_range("In NeuralNetwork::getWeight");
  return network->at(layer)->at(unit).getWeight(index);
} // End method getWeight

/**
 * Metod getNumberOfInputs
 *
 * Restituisce il numero di inputs della rete neurale
 */
uint NeuralNetwork::getNumberOfInputs() const {
  return inputs.size();
} // End method getNumberOfInputs

/**
 * Metod getNumberOfOutputs
 *
 * Restituisce il numero di outputs della rete neurale
 */
uint NeuralNetwork::getNumberOfOutputs() const {
  return network->back()->size();
} // End method getNumberOfOutputs

/**
 * Metod getNumberOfUnits
 *
 * Restituisce il numero di unita` presenti nell'i-esimo strato della rete
 */
uint NeuralNetwork::getNumberOfUnits(uint i) const {
  if (i >= nlayers)
      throw std::out_of_range("In NeuralNetwork::getNumberOfUnits");
  return network->at(i)->size();
} // End method getNumberOfUnits

/**
 * Metod getNumberOfUnits
 *
 * Restituisce il numero di unita` presenti nella rete neurale
 */
uint NeuralNetwork::getNumberOfUnits() const {
  uint n = 0;
  for (uint i = 0; i < nlayers; ++i) n += network->at(i)->size();
  return n;
} // End method getNumberOfUnits

/**
 * Metod getNumberOfHiddenUnits
 *
 * Restituisce il numero di unita` nascoste presenti nella rete neurale
 */
uint NeuralNetwork::getNumberOfHiddenUnits() const {
  return getNumberOfUnits() - getNumberOfOutputs();
} // End method getNumberOfHiddenUnits

/**
 * Metod getNumberOfWeight
 *
 * Restituisce il numero di pesi dell'unita` identificata dai parametri:
 *   - layer  : indice dello strato
 *   - unit   : indice dell'unita` nello strato scelto
 */
uint NeuralNetwork::getNumberOfWeight(uint layer, uint unit) const {
  if (layer >= nlayers || unit >= network->at(layer)->size() )
    throw std::out_of_range("In NeuralNetwork::getNumberOfWeight");
  return network->at(layer)->at(unit).getNumberOfWeights();
} // End method getNumberOfWeight

/**
 * Metod getNumberOfLayers
 *
 * Restituisce il numero di strati della rete neurale (nascosti + quello di
 * output)
 */
uint NeuralNetwork::getNumberOfLayers() const {
  return nlayers;
} // End method getNumberOfLayers

/**
 * Metod getNumberOfHiddenLayers
 *
 * Restituisce il numero di strati nascosti della rete neurale
 */
uint NeuralNetwork::getNumberOfHiddenLayers() const {
  return getNumberOfLayers() - 1;
} // End method getNumberOfHiddenLayers

/**
 * Metod getLayerDimension
 *
 * Restituisce la dimensione dell'i-esimo strato
 */
uint NeuralNetwork::getLayerDimension(uint i) const {
  if (i >= nlayers)
    throw std::out_of_range("In NeuralNetwork::getLayerDimension");
  return network->at(i)->size();
} // End method getLayerDimension

/**
 * Metod sumToWeight
 *
 * Somma il valore (value) passato come parametro al valore precedente del peso
 * specificato attraverso i seguenti parametri:
 *   - layer  : indice dello strato
 *   - unit   : indice dell'unita` nello strato scelto
 *   - index  : indice del peso nell'unita` scelta
 */
void NeuralNetwork::sumToWeight(uint layer, uint unit, uint index, real value) {
  if (layer >= nlayers || unit >= network->at(layer)->size() ||
      index >= network->at(layer)->at(unit).getNumberOfWeights() )
    throw std::out_of_range("In NeuralNetwork::sumToWeight");
  network->at(layer)->at(unit).sumToWeight(index, value);
  return;
} // End method sumToWeight

/**
 * Method compute
 *
 * Calcola l'output della rete neurale a partire dagli ultimi inputs inseriti.
 * Con il metodo getOutput e` quindi possibile accedere all'output calcolato
 */
void NeuralNetwork::compute() {
  // imposta gli inputs nelle unita` del primo strato nascosto
  for (uint i = 0; i < ninputs; ++i)
    for (uint j = 0; j < network->at(0)->size(); ++j)
      network->at(0)->at(j).setInput(i+1, inputs[i]);
  // calcola gli output degli strati nascosti successivi
  for (uint i = 0; i < nlayers-1; ++i)
    for (uint j = 0; j < network->at(i)->size(); ++j)
      for (uint t = 0; t < network->at(i+1)->size(); ++t)
        network->at(i+1)->at(t).setInput( j+1, network->at(i)->at(j).
            computeOutput() );
  // calcola l'output dello strato di output (ultimo strato)
  for (uint i = 0; i < network->at(nlayers-1)->size(); ++i)
    lastOutput[i] = network->at(nlayers-1)->at(i).computeOutput();
  return;
} // End method compute

/**
 * Method write
 *
 * Scrive l'oggetto sullo stream passato come parametro. La rete viene stampata
 * secondo il seguente formato:
 *   # number of inputs
 *   ninputs
 *   # number of layers
 *   nlayers
 *   # units for any layer
 *   nunits(1),nunits(2),...,nunits(n)
 *   # units layer 1
 *   unit(1,1)
 *   unit(1,2)
 *   ...
 *   # units layer n
 *   unit(n,1)
 *   unit(n,2)
 *   ...
 */
const NeuralNetwork& NeuralNetwork::write(std::ostream& os) const {
  os <<"# number of inputs" <<std::endl;
  os <<ninputs <<std::endl;
  os <<"# number of layers" <<std::endl;
  os <<network->size() <<std::endl;
  os <<"# units for any layer" <<std::endl;
  os <<network->at(0)->size();
  for (uint i = 1; i < network->size(); ++i)
    os <<',' <<network->at(i)->size();
  os <<std::endl;
  for (uint i = 0; i < network->size(); ++i) {
    os <<"# units layer " <<i <<std::endl;
    for (uint j = 0; j < network->at(i)->size(); ++j)
      os <<network->at(i)->at(j) <<std::endl;
  }
  return *this;
} // End method write

/**
 * Method read
 *
 * Legge l'oggetto dallo stream passato come parametro. La rete viene letta
 * secondo il seguente formato:
 *   # number of inputs
 *   ninputs
 *   # number of layers
 *   nlayers
 *   # units for any layer
 *   nunits(1),nunits(2),...,nunits(n)
 *   # units layer 1
 *   unit(1,1)
 *   unit(1,2)
 *   ...
 *   # units layer n
 *   unit(n,1)
 *   unit(n,2)
 *   ...
 * Vengono ignorate le righe che iniziano con #.
 */
NeuralNetwork& NeuralNetwork::read(std::istream& is) {
  // legge i valori dallo stream
  std::string line;
  uint ninputs, nlayers;
  std::vector<std::string>* nunitsv;
  std::vector< std::vector<Unit>* >* network;
  // legge il numero di input (# number of inputs)
  if (!readNextGoodLine(is, line)) throw read_error("In NeuralNetwork::read");
  ninputs = Global::toUint(line);
  // legge il numero di strati (# number of layers)
  if (!readNextGoodLine(is, line)) throw read_error("In NeuralNetwork::read");
  nlayers = Global::toUint(line);
  // legge il numero di unita` per ogni strato (# units for any layer)
  if (!readNextGoodLine(is, line)) throw read_error("In NeuralNetwork::read");
  nunitsv = Global::split(line,',');
  if (nunitsv->size() != nlayers) throw read_error("In NeuralNetwork::read");
  // legge le unita` dello strato i-esimo (# units layer i)
  network = new std::vector< std::vector<Unit>* >(nlayers, NULL);
  for (uint i = 0; i < nlayers; ++i) {
    uint nunits = Global::toUint(nunitsv->at(i));
    network->at(i) = new std::vector<Unit>(nunits);
    // legge l'unita` j-esima (unit(i,j))
    for (uint j = 0; j < nunits; ++j) {
      if (!readNextGoodLine(is, line))
        throw read_error("In NeuralNetwork::read");
      std::istringstream unitiss(line);
      unitiss >>network->at(i)->at(j);
    } // end for j
  } // end for i
  delete nunitsv;
  // aggiorna la rete con i parametri letti
  this->ninputs = ninputs;
  this->nlayers = nlayers;
  this->inputs.clear();
  this->inputs.resize(ninputs, 0.0);
  this->network = network;
  this->lastOutput.clear();
  this->lastOutput.resize(this->network->back()->size() , 0.0);
  return *this;
} // End method read

/**
 * Method saveOnFile
 *
 * Salva la rete neurale sul file con nome passato come parametro. Se il file
 * esiste gia` viene sovrascritto. Per il formato vedere il metodo write
 */
void NeuralNetwork::saveOnFile(const std::string& filename) const {
  std::ofstream ofs(filename.c_str());
  if (!ofs.is_open()) throw file_error("In NeuralNetwork::saveOnFile");
  this->write(ofs);
  ofs.close();
  return;
} // End method saveOnFile

// ===============
// PRIVATE METHODS
// ===============

/**
 * Method readNextGoodLine
 *
 * Legge dallo stream is una riga, controlla che non ci siano errori, non sia
 * vuota o un commento (inizia con #) e la mette in line.
 * Se trova una riga "buona" restituisce true, altrimenti restituisce false.
 */
bool NeuralNetwork::readNextGoodLine(std::istream& is, std::string& line) {
  std::string tmpline;
  std::getline(is, tmpline);
  while (is.good()) {
    Global::trim(line," \t");
    if (!tmpline.empty() && tmpline[0] != '#') {
      line = tmpline;
      return true;
    }
    std::getline(is, tmpline);
  } // end while (is.good())
  return false;
} // End method readNextGoodLine

// =================
// RELATED FUNCTIONS
// =================

/**
 * Function operator<<
 *
 * Stampa la rete neurale sullo stream out (passato come parametro). Per il
 * formato di scrittura vedere il metodo write
 */
std::ostream& operator<<(std::ostream& os, const NeuralNetwork& nn) {
  nn.write(os);
  return os;
} // End function operator<<

/**
 * Function operator>>
 *
 * Lagge la rete neurale dallo stream di input (passato come parametro). Per il
 * formato di lettura vedere il metodo read
 */
std::istream& operator>>(std::istream& is, NeuralNetwork& nn) {
  nn.read(is);
  return is;
} // End function operator<<
