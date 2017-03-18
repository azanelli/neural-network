#include "trainer.h"

#include <iostream>
#include <fstream>
#include <limits>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include "global.h"
#include "exception.h"
#include "neuralnetwork.h"
#include "backpropagation.h"
#include "dataset.h"

typedef Global::uint uint;
typedef Global::real real;

/**
 * Constructor Trainer
 *
 * Costruisce un oggetto di tipo Trainer, con modello e algoritmo di training
 * come parametri passati al costruttore.
 */
Trainer::Trainer(NeuralNetwork* model, BackPropagation* algorithm) :
    model(model),
    initmodel(new NeuralNetwork(*model)),
    algorithm(algorithm),
    epochs(0), maxepochs(0), shfepochs(0),
    vaerr(0.0), trerr(0.0), stoperr(-1),
    vaacc(0.0), tracc(0.0), stopacc(1.1),
    threshold(0.5),
    prevtrerr(0.0),
    stoperrch_var(0.0),
    stoperrch_ep(0), stoperrch_n(0)
{ } // End constructor

/**
 * Destructor ~Trainer
 */
Trainer::~Trainer() {
  delete initmodel;
}

// ==============
// PUBLIC METHODS
// ==============

/**
 * Method setDataSet
 *
 * Carica il dataset dal file passato come parametro. Il numero di inputs e il
 * numero di outputs viene stabilito dal model impostato. Il file dev'essere in
 * formato csv con i seguenti campi:
 *   id, x1, ..., xn, y1, ..., ym
 * per n inputs ed m outputs.
 */
void Trainer::setDataSet(const std::string& filename) {
  assert( model != NULL );
  dataset.load(
      filename, model->getNumberOfInputs(), model->getNumberOfOutputs() );
  return;
} // End method setDataSet

/**
 * Method setFolds
 *
 * Imposta il numero di partizioni (folds) in cui dividere il dataset. Ad ogni
 * invocazione di questo metodo vengono create n partizioni casuali differenti
 * (vengono riordinati in modo casuale i dati del dataset il quale viene poi
 * diviso in n folds). Se impostato a 1 viene fatto training sull'intero dataset
 * (permutato in modo casuale), senza validation.
 */
void Trainer::setFolds(uint n) {
  dataset.randomShuffle();
  dataset.setFolds(n);
  return;
} // setFolds

/**
 * Method setFolds
 *
 * Imposta la partizione su cui fare validation. L'indice k deve avere valore
 * 0 <= k < folds, dove folds e` il valore impostato con il metodo setFolds (il
 * numero di partizioni).
 */
void Trainer::setValidationOn(uint k) {
  dataset.setValidationFold(k);
} // setNumberOfFolds

/**
 * Method setMaxEpochs
 *
 * Imposta il numero massimo di epoche dopo la quale l'alogritmo di training si
 * ferma. Se impostato a 0 il valore e` considerato infinito (continua finche`
 * non e` verificato un altro criterio di stop).
 */
void Trainer::setMaxEpochs(uint value) {
  this->maxepochs = value;
} // End method setMaxEpochs

/**
 * Method setShuffleEpochs
 *
 * Imposta il numero epoche ogni cui riordinare in modo casuale le istanze nel
 * training set. Con v >= 1, ogni v epoche l'algoritmo di training ordina in
 * modo casuale le istanze nel training set (non quelle nel validation set che
 * rimangono invariate). Se v = 1 ad ogni epoca viene creato un riordinamento
 * casuale. Se v = 0 il training set non viene mai riordinato.
 * Il valore v dev'essere minore/uguale del numero massimo di epoche impostato.
 */
void Trainer::setShuffleEpochs(uint v) {
  this->shfepochs = v;
} // End method setShuffleEpochs

/**
 * Method setStopError
 *
 * Il processo di training si ferma quando l'errore sul training set e`
 * minore/uguale del valore passato come parametro (error). Il valore di error
 * dev'essere un numero positivo, se il parametro error e` negativo allora il
 * training termina solo quando si verifica un altro criterio di stop.
 */
void Trainer::setStopError(real error) {
  this->stoperr = error;
} // End method setStopError

/**
 * Method setStopErrorChange
 *
 * Il processo di training si ferma quando l'errore sul training set
 * varia meno della percentuale impostata con il parametro variation per un
 * numero di epoche impostato con il parametro epochs. Per esempio: per fermare
 * il training quando l'errore varia meno dello 0.1% per 10 epoche consecutive
 * si devono passare i parametri 0.1 e 10 rispettivamente in variation ed
 * epochs. Se il parametro epochs e` impostato a zero il training si ferma
 * quando si verifica un altro criterio di stop.
 */
void Trainer::setStopErrorChange(float variation, uint epochs) {
  this->stoperrch_var = variation;
  this->stoperrch_ep = epochs;
} // End method setStopErrorChange

/**
 * Method setStopAccuracy
 *
 * Il processo di training si ferma quando il valore di accuratezza sul
 * training set e` maggiore/uguale del valore passato come parametro.
 * Il valore del parametro accuracy dev'essere un valore compreso tra 0 e 1
 * (che rappresenta la percentuale di accuratezza), se il valore e` maggiore di
 * uno il training termina solo quando si verifica un altro criterio di stop.
 */
void Trainer::setStopAccuracy(real accuracy) {
  this->stopacc = accuracy;
} // End method setStopAccuracy

/**
 * Method setThreshold
 *
 * Imposta la soglia per la classificazione. Un output e` considerato corretto
 * se l'output nel dataset e il rispettivo output del modello sono entrambi
 * maggiori della soglia impostata, oppure entrambi minori. In termini piu`
 * precisi: dato il j-esimo output d(j) del dataset e il corrispettivo output
 * y(j) del modello e la soglia impostata t, allora y(j) e` considerato
 * corretto se
 *   (d(j) <= t and y(j) <= t) or (d(j) > t and y(j) > t)
 * L'output del modello e` corretto se e solo se tutti i suoi output sono
 * corretti per la formula sopra. Questo e` utilizzato nel calcolo
 * dell'accuratezza.
 * Il valore del parametro threshold dev'essere un valore nell'intervallo [0,1].
 */
void Trainer::setThreshold(real threshold) {
  assert(threshold >= 0 && threshold <= 1);
  this->threshold = threshold;
} // End method setThreshold

/**
 * Method setSaveResults
 *
 * Imposta il file in cui salvare i risultati del training. Se la stringa
 * passata e` una stringa vuota i dati non vengono salvati.
 * I dati vengono salvati nel seguente formato (csv):
 *   epoch, tr.error, va.error, tr.accuracy, va.accuracy
 * con una riga per ogni epoca del training.
 */
void Trainer::setSaveResults(const std::string& file) {
  resfile = file;
  if (!resfile.empty()) {
    std::ofstream ofs(resfile.c_str());
    if (!ofs.is_open()) throw file_error("In Trainer::setSaveResults");
    ofs <<"\"epoch\",\"tr_error\",\"va_error\",\"tr_accuracy\",\"va_accuracy\"";
    ofs <<std::endl;
    ofs.close();
  }
  return;
} // End method setSaveResults

/**
 * Method resetModel
 *
 * Ripristina il modello a quello di partenza, come se il training non fosse
 * avvenuto.
 */
void Trainer::resetModel ( ) {
  delete model;
  model = new NeuralNetwork(*initmodel);
} // End method resetModel

/**
 * Method getEpochs
 *
 * Restituisce il numero di epoche utilizzate nell'ultima sessione di training
 * (avvenuta dopo aver invocato il metodo start).
 */
uint Trainer::getEpochs() const {
  return epochs;
} // End method getEpochs

/**
 * Method getTrainingError
 *
 * Restituisce l'errore finale di training dopo l'ultimo processo di training
 * (avviato con il metodo start). L'errore restituito e` l'errore quadratico
 * medio del modello sul dataset di training, durante l'ultima epoca eseguita;
 * calcolato nel seguente modo
 *   E := 0;
 *   per ogni elemento nel training set:
 *     E += (1/2) Sum( (d(j)-y(j))^2 ) ,
 *     con d output nel dataset e y output del modello;
 *   E := E / N , con N numero di elementi nel training set;
 */
real Trainer::getTrainingError() const {
  return trerr;
} // End method getTrainingError

/**
 * Method getValidationError
 *
 * Restituisce l'errore finale di validation dopo l'ultimo processo di training
 * (avviato con il metodo start). L'errore restituito e` l'errore quadratico
 * medio del modello, sul dataset di validation, durante l'ultima epoca
 * eseguita; calcolato nel seguente modo
 *   E := 0;
 *   per ogni elemento nel validation set:
 *     E += (1/2) Sum( (d(j)-y(j))^2 ) ,
 *     con d output nel dataset e y output del modello;
 *   E := E / N , con N numero di elementi nel validation set;
 */
real Trainer::getValidationError() const {
  return vaerr;
} // End method getValidationError

/**
 * Method getTrainingAccuracy
 *
 * Restituisce l'accuratezza sul dataset di training durante l'ultimo processo
 * di apprendimento (avviato con il metodo start). L'accuratezza e` il rapporto
 * tra il numero di output predetti in modo corretto e il numero totale di
 * istanze (nel training set). Da notare che se il modello ha piu` di un output
 * la risposta e` considerata corretta se e` corretta per ogni output.
 */
real Trainer::getTrainingAccuracy ( ) const {
  return tracc;
} // End method getTrainingAccuracy

/**
 * Method getValidationAccuracy
 *
 * Restituisce l'accuratezza sul dataset per la validation durante l'ultimo
 * processo di training (avviato con il metodo start). L'accuratezza e` il
 * rapporto tra il numero di output predetti in modo corretto e il numero totale
 * di istanze (nel validation set). Da notare che se il modello ha piu` di un
 * output la risposta e` considerata corretta se e` corretta per ogni output.
 */
real Trainer::getValidationAccuracy ( ) const {
  return vaacc;
} // End method getValidationAccuracy

/**
 * Method getMinTrainingError
 *
 * Restituisce una coppia contenente il minimo errore di training raggiunto
 * durante l'ultimo processo di training e l'epoca a cui tale limite e` stato
 * raggiunto.
 */
const std::pair<real, uint>& Trainer::getMinTrainingError() const {
  return mintrerr;
} // end method getMinTrainingError

/**
 * Method getMinValidationError
 *
 * Restituisce una coppia contenente il minimo errore di validation raggiunto
 * durante l'ultimo processo di training e l'epoca a cui tale limite e` stato
 * raggiunto.
 */
const std::pair<real, uint>& Trainer::getMinValidationError() const {
  return minvaerr;
} // end method getMinValidationError

/**
 * Method getValidationAccuracy
 *
 * Restituisce una coppia contenente la massima accuracy raggiunta sul training
 * set (durante l'ultimo processo di training) e l'epoca a cui tale limite e`
 * stato raggiunto.
 */
const std::pair<real, uint>& Trainer::getMaxTrainingAccuracy() const {
  return maxtracc;
} // end method getMinTrainingError

/**
 * Method getValidationAccuracy
 *
 * Restituisce una coppia contenente la massima accuracy raggiunta sul
 * validation set (durante l'ultimo processo di training) e l'epoca a cui tale
 * limite e` stato raggiunto.
 */
const std::pair<real, uint>& Trainer::getMaxValidationAccuracy() const {
  return maxvaacc;
} // end method getMinTrainingError

/**
 * Method getFolds
 *
 * Restituisce il numero di folds impostati.
 */
uint Trainer::getFolds() const {
  return dataset.getFolds();
} // End method getValidationAccuracy

/**
 * Method getFoldDimension
 *
 * Restituisce il numero di istanze dell'i-esimo fold.
 */
uint Trainer::getFoldDimension(uint i) const {
  return dataset.getFoldSize(i);
} // End method getFoldDimension

/**
 * Method getDatasetDimension
 *
 * Restituisce il numero di istanze presenti nel dataset caricato.
 */
uint Trainer::getDatasetDimension ( ) const {
  return dataset.getSize();
} // End method getDatasetDimension

/**
 * Method start
 *
 * Esegue l'algoritmo di training sul modello, fermandosi dopo aver raggiunto un
 * criterio di stop impostato o il numero massimo di epoche.
 * Prima di eseguire questo metodo assicurarsi di aver impostato tutti i
 * parametri con il relativi metodi (in particolare di aver impostato un
 * dataset).
 */
void Trainer::start() {
  assert( model != NULL && algorithm != NULL );
  // imposta il modello nell'algoritmo di training
  algorithm->setModel(model);
  // azzera le variabili
  resetTrainingVariables();
  // ripete per ogni epoca il training
  for (epochs = 0; (epochs < maxepochs || maxepochs == 0); ++epochs) {
    // crea un ordine casuale delle istanze del training set
    if ( (shfepochs != 0) && (epochs % shfepochs == 0) )
      dataset.randomShuffleTrainingSet();
    // esegue training e validation sul dataset
    training();
    validation();
    // aggiorna le variabili globali
    updateTrainingVariables();
    // salva i risultati di questa epoca
    saveEpochResults();
    // controlla il criterio di stop impostato
    if (checkStop()) break;
  } // end for epochs
  return;
} // End method start

// ===============
// PRIVATE METHODS
// ===============

/**
 * Method training
 *
 * Applica l'algoritmo di training sul training set.
 * Aggiorna i valori per le seguenti variabili con gli errori di training:
 *   - trerr : errore quadratico medio di training
 *   - tracc : accuracy sul dataset di training
 */
void Trainer::training() {
  // azzerra le variabili
  trerr = 0.0;
  tracc = 0.0;
  for (uint element = 0; element < dataset.getTrSetSize(); ++element) {
    // esegue l'algoritmo su un elemento del dataset
    algorithm->compute(
        dataset.trAt(element).input, dataset.trAt(element).output );
    // calcola i nuovi errori
    model->setInputs(dataset.trAt(element).input);
    model->compute();
    trerr += modelError(model->getOutputs(), dataset.trAt(element).output);
    tracc += modelHit(model->getOutputs(), dataset.trAt(element).output);
  } // end for element
  trerr = trerr / (real(dataset.getTrSetSize()));
  tracc = tracc / (real(dataset.getTrSetSize()));
  return;
} // End method training

/**
 * Method validation
 *
 * Esegue la validazione sulla partizione del dataset impostata.
 * Aggiorna i valori delle seguenti variabili con gli errori di validation:
 *   - vaerr : errore quadratico medio di validation
 *   - vaacc : accuracy (percentuale) sul dataset di validation
 */
void Trainer::validation() {
  // azzera le variabili
  vaerr = 0.0;
  vaacc = 0.0;
  // se il validation set e` vuoto non fa nulla
  if (dataset.getVaSetSize() == 0) return;
  // per ogni elemento della partizione
  for (uint element = 0; element < dataset.getVaSetSize(); ++element) {
    model->setInputs(dataset.vaAt(element).input);
    model->compute();
    vaerr += modelError(model->getOutputs(), dataset.vaAt(element).output);
    vaacc += modelHit(model->getOutputs(), dataset.vaAt(element).output);
  }
  vaerr = vaerr / real(dataset.getVaSetSize());
  vaacc = vaacc / real(dataset.getVaSetSize());
  return;
} // End method validation

/**
 * Method modelError
 *
 * Prende gli outputs del modello e quelli del dataset e restituisce l'errore
 * che il modello ha rispetto agli outputs del dataset.
 * L'errore e` calcolato secondo la formula
 *   E = (1/2) * Sum( (d(j)-y(j))^2 )
 * dove j e` l'indice degli outputs (su cui viene fatta la somma), d(j) e` il
 * valore del j-esimo output nel dataset e y(j) e` il j-esimo output del
 * modello.
 */
inline
real Trainer::modelError(const std::vector<real>& mout,
    const std::vector<real>& dsout) const {
  assert(mout.size() == dsout.size());
  real error = 0.0;
  for (uint i = 0; i < mout.size(); ++i)
    error += pow(dsout[i] - mout[i], 2);
  return error / 2;
} // end method modelErrorOn

/**
 * Method modelHit
 *
 * Restituisce 1 se l'output mout (output del modello) e` corretto rispetto
 * all'output dsout (output nel dataset), 0 altrimenti. Dato il j-esimo output
 * del dataset d(j) e il corrispettivo output del modello y(j) e la soglia
 * impostata t, allora y(j) e` considerato corretto se
 *   (d(j) <= t and y(j) <= t) or (d(j) > t and y(j) > t)
 * L'output del modello e` corretto se e solo se tutti i suoi output sono
 * corretti per la formula sopra.
 */
inline
uint Trainer::modelHit(const std::vector<real>& mout,
    const std::vector<real>& dsout) const {
  assert(mout.size() == dsout.size());
  for (uint i = 0; i < mout.size(); ++i)
    if ( ((dsout[i] > threshold) && (mout[i] <= threshold)) ||
         ((dsout[i] <= threshold) && (mout[i] > threshold)) )
      return false;
  return true;
} // End method modelHit

/**
 * Method resetTrainingVariables
 *
 * Azzera le seguenti variabili globali utilizzate nell'algoritmo di training
 * (vedere il metodo start):
 *   - mintrerr
 *   - minvaerr
 *   - maxtracc
 *   - maxvaacc
 *   - prevtrerr
 *   - n_stoperrch
 */
inline
void Trainer::resetTrainingVariables() {
  mintrerr = std::make_pair(std::numeric_limits<real>::infinity(), 0);
  minvaerr = std::make_pair(std::numeric_limits<real>::infinity(), 0);
  maxtracc = std::make_pair(0.0, 0);
  maxvaacc = std::make_pair(0.0, 0);
  prevtrerr = 0;
  stoperrch_n = 0;
  return;
} // End method resetTrainingVariables

/**
 * Method updateTrainingVariables
 *
 * Aggiorna le seguenti variabili globali al termine di ogni epoca della
 * procedura di training (vedere il metodo start):
 *   - mintrerr
 *   - minvaerr
 *   - maxtracc
 *   - maxvaacc
 */
inline
void Trainer::updateTrainingVariables() {
  if (trerr < mintrerr.first) {
    mintrerr.first = trerr;
    mintrerr.second = epochs;
  }
  if (vaerr < minvaerr.first) {
    minvaerr.first = vaerr;
    minvaerr.second = epochs;
  }
  if (tracc > maxtracc.first) {
    maxtracc.first = tracc;
    maxtracc.second = epochs;
  }
  if (vaacc > maxvaacc.first) {
    maxvaacc.first = vaacc;
    maxvaacc.second = epochs;
  }
  return;
} // End method updateTrainingVariables

/**
 * Method checkStop
 *
 * Verifica se si e` raggiunto il criterio di stop impostato, restituisce
 * true se si ci puo` fermare, false altrimenti.
 */
inline
bool Trainer::checkStop() {
  // il training ha portato a divergenza (con risultati fuori dai limiti)
  if (std::isinf(trerr) || std::isnan(trerr)) return true;
  if (std::isinf(vaerr) || std::isnan(vaerr)) return true;
  if (trerr < 0 || vaerr < 0 || tracc < 0 || vaacc < 0) return true;
  // si e` raggiunto l'errore minimo impostato
  if (trerr <= stoperr) return true;
  // si e` raggiunta l'accuratezza massima impostata
  if (tracc >= stopacc) return true;
  // controlla la variazione di errore
  if (checkStopErrorChange()) return true;
  // nessuna soglia di stop e` stata raggiunta
  return false;
} // End method checkStop

/**
 * Method checkStopErrorChange
 *
 * Verifica se per un numero di epoche impostato nell'attributo epocherrch_ep
 * il cambiamento dell'errore e` stato minore della percentuale nell'attributo
 * stoperrch_var, in tal caso restituisce true, altrimenti restituisce false.
 */
inline
bool Trainer::checkStopErrorChange() {
  if (stoperrch_ep == 0) return false;
  if ( fabs((trerr-prevtrerr)/trerr) <= (stoperrch_var/100.0) )
    ++stoperrch_n;
  else
    stoperrch_n = 0;
  prevtrerr = trerr;
  if (stoperrch_n >= stoperrch_ep) return true;
  return false;
} // End method checkStopErrorChange

/**
 * Method saveEpochResults
 *
 * Appende una riga nel file "resfile" con il seguente formato:
 *   epochs, trerr, vaerr, tracc, vaacc
 * prendendo i valori dalle relative variabili
 */
void Trainer::saveEpochResults() const {
  if (resfile.empty()) return;
  std::ofstream ofs;
  // modifica il formato di stampa per i numeri floating point
  std::streamsize prprec = ofs.precision(5);
  std::ios::fmtflags prflag =
      ofs.setf(std::ios::scientific, std::ios::floatfield);
  // apre il file e aggiunge una riga con i risultati
  ofs.open(resfile.c_str(),  std::ios::out | std::ios::app);
  if (!ofs.is_open()) throw file_error("In Trainer::saveEpochResults");
  ofs <<(epochs+1) <<"," <<trerr <<"," <<vaerr <<"," <<tracc <<"," <<vaacc;
  ofs <<std::endl;
  ofs.close();
  // ripristina il formato stampa
  ofs.precision(prprec);
  ofs.setf(prflag, std::ios::floatfield);
  return;
} // End method saveEpochResults
