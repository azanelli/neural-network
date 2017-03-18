#ifndef DATASET_H_
#define DATASET_H_

#include <vector>
#include <string>
#include "global.h"

typedef Global::uint uint;
typedef Global::real real;

/**
 * Rappresenta un dataset di istanze della forma <id,inputs,outputs>. Con il
 * metodo load si puo` indicare il nome di un file (in formato csv) da dove
 * caricare le istanze.
 * Con altri metodi e` possibile accedere alle varie istanze (all'id, agli
 * inputs o agli outputs). Con il metodo randomShuffle si crea una permutazione
 * casuale delle istanze; con il metodo restore si ripristina l'ordine
 * originale delle istanze.
 * Con il metodo setFolds e` possibile dividere il dataset in partizioni uguali
 * e impostarne una come validation set con il metodo setValidationFold. Con i
 * metodi trAt e vaAt is puo` accedere agli elementi del training set e del
 * validation set (se creati con setFolds e setValidationFold).
 */
class Dataset
{
  public:
    Dataset();
    virtual ~Dataset() { }

    struct Instance {
      std::string id;
      std::vector<real> input;
      std::vector<real> output;
    };

    void load ( const std::string& filename, uint ninputs, uint noutputs );
    void setFolds ( uint n );
    void setValidationFold ( uint k );
    void merge ( );
    bool isEmpty ( ) const;
    uint getSize ( ) const;
    uint getFolds ( ) const;
    uint getFoldSize ( uint i ) const;
    uint getTrSetSize ( ) const;
    uint getVaSetSize ( ) const;
    const std::string& getId ( uint i ) const;
    const std::vector<real>& getInputs ( uint i ) const;
    const std::vector<real>& getOutputs ( uint i ) const;
    const Instance& trAt ( uint i );
    const Instance& vaAt ( uint i );
    const Instance& at ( uint i ) const;
    const Instance& operator[] ( uint i ) const;
    void randomShuffleTrainingSet ( );
    void randomShuffle ( );
    void restore ( );

  private:
    std::vector<Instance> dataset;
    std::vector<uint> av; // access vector
    std::vector<uint> trav; // training set access vector
    uint folds, vafold;

    void makeTrAccessVector();
    uint startIndexFold(uint k) const;
    uint endIndexFold(uint k) const;
    uint foldDimension(uint k) const;

}; // End class Dataset

#endif /* DATASET_H_ */
