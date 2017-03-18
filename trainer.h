#ifndef TRAINING_H_
#define TRAINING_H_

#include <string>
#include <vector>
#include "global.h"
#include "neuralnetwork.h"
#include "backpropagation.h"
#include "dataset.h"

typedef Global::uint uint;
typedef Global::real real;

/**
 * Class Trainer
 *
 * Dato un modello per classificazione e un algoritmo di training per tale
 * modello, si puo` costruire un oggetto Trainer che applica la procedura di
 * training al modello impostato, utilizzando come algoritmo di training
 * l'algoritmo impostato. I dati per il training vengono caricati dal dataset
 * (un file in formato csv) impostato con il metodo setDataSet.
 * La procedura di training viene compiuta in una modalita` "orientata" alla
 * k-fold cross validation: e` possibile impostare il numero di folds in cui
 * dividere il training set e selezionare quale utilizzare per la validation.
 * Per compiere una k-fold cross validation completa si deve ri-avviare il
 * processo k volte utilizzando (ciclicamente) tutti i fold per la validation.
 * Impostando a 1 il numero di folds (con il metodo setFolds) il training e`
 * fatto sull'intero dataset (senza validation), quindi gli errori di validation
 * saranno nulli.
 * E` possibile impostare piu` di un criterio di stop dove interrompere la
 * procedura di training (numero massimo di epoche, errore minimo, ecc.).
 * Con il metodo start si avvia il training; una volta terminato il training si
 * possono leggere i risultati finali con gli appositi metodi.
 */
class Trainer
{
  public:
    Trainer ( NeuralNetwork* model, BackPropagation* algorithm );
    virtual ~Trainer ( );

    void setDataSet ( const std::string& file );
    void setFolds ( uint n );
    void setValidationOn ( uint k );
    void setMaxEpochs ( uint value );
    void setShuffleEpochs ( uint v );
    void setStopError ( real error );
    void setStopErrorChange ( float variation, uint epochs );
    void setStopAccuracy ( real accuracy );
    void setThreshold ( real threshold );
    void setSaveResults ( const std::string& file );
    void resetModel ( );
    uint getEpochs ( ) const;
    real getTrainingError ( ) const;
    real getValidationError ( ) const;
    real getTrainingAccuracy ( ) const;
    real getValidationAccuracy ( ) const;
    const std::pair<real, uint>& getMinTrainingError ( ) const;
    const std::pair<real, uint>& getMinValidationError ( ) const;
    const std::pair<real, uint>& getMaxTrainingAccuracy ( ) const;
    const std::pair<real, uint>& getMaxValidationAccuracy ( ) const;
    uint getFolds ( ) const;
    uint getFoldDimension ( uint i ) const;
    uint getDatasetDimension ( ) const;
    void start ( );

  private:
    NeuralNetwork* model;
    NeuralNetwork* initmodel;
    BackPropagation* algorithm;
    Dataset dataset;
    uint epochs, maxepochs, shfepochs;
    real vaerr, trerr, stoperr;
    real vaacc, tracc, stopacc;
    std::pair<real, uint> mintrerr, minvaerr, maxtracc, maxvaacc;
    real threshold;
    real prevtrerr;
    float stoperrch_var;
    uint stoperrch_ep, stoperrch_n;
    std::string resfile;

    void training();
    void validation();
    real modelError ( const std::vector<real>& mout,
        const std::vector<real>& dsout) const;
    uint modelHit ( const std::vector<real>& mout,
        const std::vector<real>& dsout) const;
    void resetTrainingVariables ( );
    void updateTrainingVariables ( );
    bool checkStop ( );
    bool checkStopErrorChange ( );
    void saveEpochResults ( ) const;

}; // End class Trainer

#endif /* TRAINING_H_ */
