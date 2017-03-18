#ifndef NNTRAINING_H_
#define NNTRAINING_H_

#include <vector>
#include <string>
#include <sys/time.h>
#include "global.h"
#include "neuralnetwork.h"
#include "backpropagation.h"
#include "trainer.h"

typedef Global::uint uint;
typedef Global::real real;

/**
 * Utilizzando un oggetto NeuralNetwork per rappresentare una rete neurale per
 * classificazione e un oggetto BackPropagation per l'algoritmo di training,
 * utilizzando la classe Trainer esegue il training della rete neurale con
 * l'algoritmo di back-propagation.
 * I parametri su come costruire la rete neurale, l'algoritmo di
 * back-propagation e come fare il training, sono parametri globali:
 *   --inputs     numero di inputs della rete neurale.
 *   --outputs    numero di outputs della rete neurale.
 *   --hlayers    numero di strati nascosti.
 *   --units      numero di unita` per ogni strato nascosto, in una unica
 *                stringa con valori separati da virgola.
 *   --eta        training rate (eta) per l'algoritmo di back-propagation.
 *   --trfile     file contenente le istanze per il training
 * Ed i seguenti parametri opzionali:
 *   --alpha      momentum rate per l'algoritmo di back-propagation (default 0).
 *   --lambda     generalization rate per l'algoritmo di back-propagation
 *                (default 0).
 *   --trsave     salva i risultati del training dopo ogni epoca nel file
 *                specificato (se non specificato i risultati non vengono
 *                salvati); vengono creati tanti files quanti sono i folds
 *                impostati, ognuno con lo stesso nome ma con un numero diverso
 *                appeso come suffisso.
 *   --folds      numero di partizioni in cui viene diviso il dataset per il
 *                processo di cross validation (default 10); se impostato a 1
 *                il training viene fatto sull'intero dataset, senza
 *                validazione.
 *   --maxfolds   numero di partizioni utilizzate (ciclicamente) durante il
 *                training per fare validazione (di default uguale al numero di
 *                folds impostati); se per per esempio e` impostato a 1 viene
 *                utilizzata una sola partizione per fare validation,
 *                riducendosi cosi` ad un processo di simple validation.
 *   --maxepochs  numero massimo di epoche per il training (default infinito)
 *   --shuffle    numero di epoche ogni cui riordinare in modo casuale il
 *                training set (non il validation set che ovviamente rimane
 *                invariato); se 1 ad ogni epoca viene riordinato; se 0 viene
 *                riordinato solamente all'inizio; il default e` 0.
 *   --stoperr    soglia dell'errore sul training set a cui il training si
 *                ferma (default nessuna).
 *   --stopacc    soglia del valore di accuracy sul training set a cui il
 *                processo di training si ferma (default nessuna).
 *   --stoperrch  ferma il processo di training se per un certo numero di epoche
 *                consecutive (di default 10) l'errore sul dataset di  training
 *                subisce variazioni percentuali inferiori al valore passato.
 *                Ad esempio con l'opzione --stoperrch 0.1 se per 10 epoche
 *                consecutive l'errore di training non cambia piu` dello 0.1%
 *                il processo viene interrotto. Per impostare il numero di
 *                epoche consecutive utilizzare il parametro --stoperrchep.
 *   --stoperrchep  imposta il numero di epoche consecutive dopo cui
 *                interrompere il processo di training se l'errore non subisce
 *                variazioni significative (vedi --stoperrch). Se il parametro
 *                --stoperrch non e` impostato questo parametro viene ignorato.
 *   --threshold  soglia per la classificazione; dev'essere un valore nell'
 *                intervallo [0,1] (default 0.5).
 *   --nnsave     salva la rete neurale, dopo il training, nel file specificato;
 *                viene salvata una rete neurale per ogni processo di training
 *                eseguito (vedere --folds e --maxfolds).
 * Dati una serie di parametri globali, con il metodo exec e` possibile avviare
 * il processo di training.
 */
class NNTraining
{
  public:
    static int exec ( );

  private:
    static Trainer* tr;
    static NeuralNetwork* nn;
    static BackPropagation* bp;
    // parametri della rete neurale
    static uint inputs, outputs, hlayers;
    static std::vector<uint> units;
    // parametri per l'algoritmo di back-propagation
    static real eta, alpha, lambda;
    // parametri per il training
    static std::string trfile, trsave, nnsave;
    static uint folds, maxfolds;
    static uint maxepochs, shuffle;
    static real stoperr, stopacc, threshold;
    static float stoperrch;
    static uint stoperrchep;
    // tempi di calcolo
    static timeval time_start, time_end;
    static clock_t tcpu_start, tcpu_end;
    // risultati medi finali
    static uint mepochs;
    static real mtrerr, mvaerr, mtracc, mvaacc;
    static real mtrerrmin, mvaerrmin, mtraccmax, mvaaccmax;
    static double mtime, mtcpu;

    static bool checkParameters ( );
    static void printNeuralNetworkInfo ( );
    static void printBackPropagationInfo ( );
    static void updateTrainingResults ( );
    static void printTrainingInfo ( );
    static void printFinalResults ( );
    static void startTimer ( );
    static void stopTimer ( );
    static double getElapsedTime ( );
    static double getCpuUsage ( );

}; // End class NNTraining

#endif /* NNTRAINING_H_ */
