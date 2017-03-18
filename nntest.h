#ifndef NNTEST_H_
#define NNTEST_H_

#include <string>
#include "global.h"
#include "neuralnetwork.h"
#include "tester.h"

typedef Global::real real;

/**
 * Dato un file contenente una rete neurale ed un file con un dataset, tramite
 * il metodo exec viene eseguito il test della rete neurale sul dataset,
 * vengono salvate (se richiesto tramite un parametro) le risposte del modello
 * su file e vengono stampati in output vari risultati del test eseguito. Le
 * risposte vengono salvate su file nel formato csv:
 *   id,output(1),...,output(n)
 * Il test puo` essere eseguito anche in modalita` "senza output", in questo
 * caso la rete neurale non puo` essere valutata e non vengono stampati
 * risultati di test, ma possono comunque essere salvati su file le risposte del
 * modello rispetto al dataset.
 * Si aspetta i seguenti parametri globali obbligatori:
 *   --nnfile     nome del file contenente la rete neurale (prodotta nella
 *                modalita` training).
 *   --dsfile     nome del file con il dataset di test.
 * Ed i seguenti parametri opzionali:
 *   --output     flag per indicare se nel dataset e` presente l'output; se
 *                il flag non e` presente vengono letti dal dataset solo gli
 *                inputs e viene (se richiesto) prodotto il file con le
 *                risposte della rete neurale.
 *   --threshold  soglia per la classificazione; dev'essere un valore nell'
 *                intervallo [0,1] (default 0.5). questo attributo ha senso
 *                solamente se il flag --output è impostato.
 *   --tssave     salva i risultati del test (le risposte della rete neurale)
 *                nel file specificato, nella forma (csv):
 *                  id, output(1), ..., output(n)
 * Il numero di input e di output nel dataset devono essere uguali al numero di
 * input e output della rete neurale.
 */
class NNTest
{
  public:
    static int exec ( );

  private:
    static Tester* ts;
    static NeuralNetwork* nn;
    // parametri
    static bool output;
    static std::string nnfile, dsfile, tssave;
    static real threshold;

    static bool checkParameters ( );
    static void printNeuralNetworkInfo ( );
    static void printTestInfo ( );

}; // End Class NNTest

#endif /* NNTEST_H_ */
