#ifndef TESTER_H_
#define TESTER_H_

#include <string>
#include "global.h"
#include "dataset.h"
#include "neuralnetwork.h"

typedef Global::uint uint;
typedef Global::real real;

/**
 * Class Tester
 *
 * Dato un modello per la classificazione un oggetto di tipo Tester permette di
 * effettuare uno o piu` test sul modello impostato. Con il metodo setDataSet si
 * imposta il file da cui caricare il dataset per eseguire il test, tale file
 * dev'essere nel seguente formato csv:
 *   id,input(1),...,input(n),output(1),output(m)
 * per un modello con n inputs e m outputs, con una riga per ogni istanza.
 * Al modello vengono presentati tutti gli inputs del dataset e viene
 * confrontata la risposta del modello con gli outputs del dataset. Il test
 * si avvia con il metodo start; terminato il test e` possibile accedere ai
 * risultati attraverso gli altri metodi.
 * Con il metodo setSaveModelResponses si puo` indicare su quale file salvare
 * le risposte del modello per ogni istanza del dataset.
 * Il test puo` essere effettuato anche senza output nel dataset (impostando
 * withoutput = false nel costruttore), in questo caso l'unico scopo utile del
 * test e` salvare le risposte del modello su file.
 */
class Tester
{
  public:
    Tester ( NeuralNetwork* model, bool withoutput = true );
    virtual ~Tester ( );

    void setDataSet ( const std::string& file );
    void setSaveModelResponses ( const std::string& file );
    void setThreshold( real threshold );
    uint getDatasetDimension ( ) const;
    uint getNumberOfMissed ( ) const;
    uint getNumberOfHits ( ) const;
    real getAccuracy ( ) const;
    real getQuadraticError ( ) const;
    void start ( );

  private:
    NeuralNetwork* model;
    Dataset dataset;
    bool withoutput;
    uint missed, hits;
    real threshold, accuracy, error;
    std::string resfile;

    bool checkModelResponse ( uint i ) const;
    real lastModelError ( uint i ) const;
    void saveLastOutputs ( const std::string& id ) const;

}; // End class Tester

#endif /* TESTER_H_ */
