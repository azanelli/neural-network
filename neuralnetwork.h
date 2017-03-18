#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include <vector>
#include <string>
#include <ostream>
#include "global.h"
#include "unit.h"

typedef Global::uint uint;
typedef Global::real real;

/**
 * Class NeuralNetwork
 *
 * Rappresenta una rete neurale multistrato con le seguenti caratteristiche:
 *   - Genera outputs tra 0 e 1, quindi da usare per classificazione
 *   - La funzione di attivazione di tutte le unita` e` f(x) = 1/(1+e(-x))
 *   - Ogni strato e` completamente connesso allo strato successivo
 *   - Un numero arbitrario di inputs (parametro nel costruttore)
 *   - Un numero arbitrario di strati nascosti con un numero arbitrario di
 *     unita` per ogni strato (parametri nel costruttore)
 *   - Uno strato di output con un numero arbitrario di unita` (parametro nel
 *     costruttore)
 * Dato un insieme di inputs impostati con il metodo setInputs, attraverso il
 * metodo compute si calcola l'output della rete che puo` essere letto con il
 * metodo getOutputs.
 * Con i metodi read e write, oppure con i relativi operatori >> e << si puo`
 * leggere e scrivere una rete neurale (ad esempio per salvarla su file) con il
 * seguente formato:
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
 * I pesi delle unita` vengono scritti (e letti) con una precisione di 10e-21.
 */
class NeuralNetwork
{
  public:
    NeuralNetwork ( );
    NeuralNetwork ( uint ninputs, uint nlayers,
        const std::vector<uint>& nunits );
    NeuralNetwork ( const NeuralNetwork& neuralnetwork );
    virtual ~NeuralNetwork();

    void setInput ( uint i, real input );
    void setInputs ( const std::vector<real>& inputs );
    void setWeight ( uint layer, uint unit, uint index, real weight );
    real getInput ( uint i ) const;
    const std::vector<real>& getInputs ( ) const;
    const std::vector<real>& getOutputs ( ) const;
    real getOutput ( uint i ) const;
    real getUnitInput ( uint layer, uint unit, uint index ) const;
    real getUnitOutput ( uint layer, uint unit ) const;
    real getWeight ( uint layer, uint unit, uint index ) const;
    uint getNumberOfInputs ( ) const;
    uint getNumberOfOutputs ( ) const;
    uint getNumberOfUnits ( uint i ) const;
    uint getNumberOfUnits ( ) const;
    uint getNumberOfHiddenUnits ( ) const;
    uint getNumberOfWeight ( uint layer, uint unit ) const;
    uint getNumberOfLayers ( ) const;
    uint getNumberOfHiddenLayers ( ) const;
    uint getLayerDimension ( uint i ) const;
    void sumToWeight ( uint layer, uint unit, uint index, real value );
    void compute ( );
    const NeuralNetwork& write ( std::ostream& os ) const;
    NeuralNetwork& read ( std::istream& is );
    void saveOnFile ( const std::string& filename ) const;

  private:
    uint ninputs;
    uint nlayers;
    std::vector<real> inputs;
    std::vector< std::vector<Unit>* >* network;
    std::vector<real> lastOutput;

    bool readNextGoodLine( std::istream& is, std::string& line );

}; // End class NeuralNetwork

std::ostream& operator<<(std::ostream&, const NeuralNetwork&);
std::istream& operator>>(std::istream&, NeuralNetwork&);

#endif /* NEURALNETWORK_H_ */
