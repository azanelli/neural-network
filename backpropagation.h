#ifndef BACKPROPAGATION_H_
#define BACKPROPAGATION_H_

#include <vector>
#include "global.h"
#include "neuralnetwork.h"

typedef Global::uint uint;
typedef Global::real real;

/**
 * Class BackPropagation
 *
 * Un oggetto di questo tipo permette di applicare l'algoritmo di training
 * back-propagation ad una rete neurale (impostata con il metodo setModel).
 * L'algoritmo di back-propagation richiede il calcolo della derivata della
 * funzione di attivazione delle unita` della rete neurale. Si suppone che tale
 * funzione sia la seguente (funzione sigmoide):
 *   f(net) = 1 / (1 + e^(-net))
 * dove net e` la combinazione lineare degli input dell'unita` pesati con i
 * relativi pesi; sia y l'output di un'unita`, la derivata di tale funzione
 * viene quindi calcolata come:
 *   f'(net) = y * (1 - y)
 * Con il metodo compute si applica un passo dell'algoritmo alla rete neurale.
 * E` possibile impostare, con i relativi metodi, i diversi parametri dell'
 * algoritmo: il learning rate (eta), il momentum rate (alpha) e il
 * regularization rate (lambda).
 */
class BackPropagation
{
  public:
    BackPropagation ( );
    virtual ~BackPropagation ( );

    void setModel ( NeuralNetwork* neuralNetwork );
    void setLearningRate ( real eta );
    void setMomentumRate ( real alfa );
    void setRegularizationRate ( real lambda );
    real getLearningRate ( ) const;
    real getMomentumRate ( ) const;
    real getRegularizationRate ( ) const;
    void compute ( const std::vector<real>& inputs,
        const std::vector<real>& desiredResponse );

  private:
    NeuralNetwork* neuralnetwork;
    real eta, lambda, alfa;
    std::vector< std::vector<real>* >* momentumtable;

    real localGradient ( real error, real output ) const;
    void updateWeight ( uint layer, uint unit, uint weight, real eta,
        real delta, real lambda, real alfa ) const;
    void makeMomentumTable ( );
    void deleteMomentumTable ( );

}; // End class BackPropagation

#endif /* BACKPROPAGATION_H_ */
