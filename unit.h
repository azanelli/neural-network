#ifndef UNIT_H_
#define UNIT_H_

#include <ostream>
#include <string>
#include <vector>
#include "global.h"

typedef Global::uint uint;
typedef Global::real real;

/**
 * Class Unit
 *
 * Rappresenta un'unita` di una rete neurale con le seguenti caratteristiche:
 *   - Funzione di attivazione f(x) = 1/(1 + e^(-x))
 *   - Output y = f(net), dove net e` la combinazione lineare degli input
 *     impostati, pesati con i pesi dell'unita`
 *   - Un numero arbitrario di input (passato nel costruttore dell'oggetto)
 *   - Un peso per ogni input, inizializzato con un numero random in [-0.7,+0.7]
 *     escluso lo 0
 *   - Un peso w0 (bias) che ha input fisso a valore +1
 *   - Se il numero di inputs dell'unita` e` k, allora il numero di pesi e` k+1
 *     (un peso per ogni input + il peso w0).
 * Dato un insieme di inputs impostati con il metodo setInputs, attraverso il
 * metodo computeOutput calcola l'output dell'unita`, che puo` essere letto con
 * il metodo getOutput.
 * Con i metodi read e write, oppure con i relativi operatori >> e <<, si puo`
 * leggere e scrivere una unita` con il seguente formato:
 *   nweights, weight(1), ..., weight(n)
 * i pesi vengono scritti (e letti) con precisione 10e^-21.
 */
class Unit
{
  public:
    Unit ( uint numberOfInputs = 0);
    virtual ~Unit ( );
    Unit ( const Unit& unit );

    void setInput ( uint i, real input );
    void setInputs ( const std::vector<real>& inputs );
    void setWeight ( uint i, real weight );
    void setWeights ( const std::vector<real>& weights );
    void sumToWeight ( uint i, real value );
    real getLastInput ( uint i ) const;
    real getLastOutput ( ) const;
    uint getNumberOfInputs ( ) const;
    uint getNumberOfWeights ( ) const;
    real getWeight ( uint i ) const;
    real computeOutput ( );
    const Unit& write ( std::ostream& os ) const;
    Unit& read ( std::istream& is );

  private:
    uint numberOfInputs;
    uint numberOfWeights;
    std::vector<real> inputs;
    std::vector<real> weights;
    real lastOutput;

    void initWeightsRandom();
    static void setRandomValue(real& val);
    real calcOutput() const;
    real activationFunction(real net) const;
}; // End class Unit

std::ostream& operator<<(std::ostream&, const Unit&);
std::istream& operator>>(std::istream&, Unit&);

#endif /* UNIT_H_ */
