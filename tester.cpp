#include "tester.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cassert>
#include "global.h"
#include "exception.h"
#include "dataset.h"
#include "neuralnetwork.h"

/**
 * Constructor Trainer
 *
 * Costruisce un oggetto Trainer con il modello (classificatore) passato come
 * paraemtro. Attraverso il parametro withoutput si specifica se il test viene
 * eseguito con o senza output nel dataset.
 */
Tester::Tester(NeuralNetwork* model, bool withoutput) :
    model(model),
    withoutput(withoutput),
    missed(0),
    hits(0),
    threshold(0.5),
    accuracy(0.0),
    error(0.0)
{ } // End constructor

/**
 * Destructor ~Trainer
 */
Tester::~Tester() { }

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
 * per n inputs ed m outputs, con un'istanza per ogni riga.
 */
void Tester::setDataSet(const std::string& filename) {
  assert( model != NULL );
  uint ninputs = model->getNumberOfInputs();
  uint noutputs = model->getNumberOfOutputs();
  if (withoutput)
    dataset.load(filename, ninputs, noutputs);
  else
    dataset.load(filename, ninputs, 0);
  return;
} // End method setDataSet

/**
 * Method setSaveModelResponses
 *
 * Durante il test salva le risposte del modello, per ogni istanza del dataset,
 * nel file passato, con il seguente formato (csv):
 *   id, output(1), ..., output(n)
 * gli output vengono salvati in formato scentifico con 5 decimali di
 * precisione. Se la stringa passata e` vuota allora le risposte non vengono
 * salvate.
 */
void Tester::setSaveModelResponses(const std::string& file) {
  resfile = file;
  // scrive l'intestazione nel file da salvare
  if (!resfile.empty()) {
    std::ofstream ofs(resfile.c_str());
    if (!ofs.is_open()) throw file_error("In Tester::setSaveModelResponse");
    ofs <<"\"id\"";
    for (uint i = 0; i < model->getNumberOfOutputs(); ++i)
      ofs <<",\"out[" <<i <<"]\"";
    ofs <<std::endl;
    ofs.close();
  }
  return;
} // End method setSaveResults

/**
 * Method setThreshold
 *
 * Imposta la soglia per la classificazione. Dato il j-esimo output d(j) del
 * dataset e il corrispettivo output y(j) del modello e la soglia impostata t,
 * allora y(j) e` considerato corretto se
 *   (d(j) <= t and y(j) <= t) or (d(j) > t and y(j) > t)
 * L'output del modello e` corretto se e solo se tutti i suoi output sono
 * corretti per la formula sopra. Questo e` utilizzato nel calcolo dell'
 * accuratezza. Il valore del parametro dev'essere compreso nell'intervallo
 * [0,1].
 */
void Tester::setThreshold(real threshold) {
  assert(threshold >= 0 && threshold <= 1);
  this->threshold = threshold;
} // End method setThreshold

/**
 * Method getDatasetDimension
 *
 * Restituisce la dimensione del dataset (il numero di istanze).
 */
uint Tester::getDatasetDimension() const {
  return dataset.getSize();
} // End method getDatasetDimension

/**
 * Method getNumberOfMissed
 *
 * Restituisce il numero di risposte errate del modello durante l'ultimo test
 * (avviato con il metodo start).
 */
uint Tester::getNumberOfMissed() const {
  return missed;
} // End method getNumberOfMissed

/**
 * Method getNumberOfHits
 *
 * Restituisce il numero di risposte corrette del modello durante l'ultimo test
 * (avviato con il metodo start).
 */
uint Tester::getNumberOfHits() const {
  return hits;
} // End method getNumberOfHits

/**
 * Method getAccuracy
 *
 * Restituisce il valore dell'accuratezza del modello durante l'ultimo test
 * (avviato con il metodo start), calcolato come il numero di risposte corrette
 * diviso il numero totale di elementi del dataset. Da notare che se il modello
 * ha piu` di un output la risposta e` considerata corretta solo se e` corretta
 * per tutti gli outputs.
 */
real Tester::getAccuracy() const {
  return accuracy;
} // End method getAccuracy

/**
 * Method getQuadraticError
 *
 * Restituisce l'errore quadratico medio del modello durante l'ultimo test
 * (avviato con il metodo start), calcolato come
 *   E := 0;
 *   per ogni elemento nel dataset:
 *     E += (1/2) Sum( (d(j)-y(j))^2 ) ,
 *     con d output nel dataset e y output del modello;
 *   E := E / N , con N numero di elementi nel dataset;
 */
real Tester::getQuadraticError() const {
  return error;
} // End method getQuadraticError

/**
 * Method start
 *
 * Avvia il test del modello sul dataset (impostato con il metodo setDataSet).
 * Se impostato il salvataggio dei risultati (con il metodo setSaveResult)
 * vengono salvati su file le risposte del modello per ogni istanza del dataset.
 * Al termine dell'esecuzione di questo metodo e` possibile accedere ai vari
 * risultati del test attraverso gli altri metodi (accuratezza, errore, ecc.).
 */
void Tester::start() {
  assert( model != NULL && !dataset.isEmpty() );
  assert( model->getNumberOfInputs() == dataset.getInputs(0).size() );
  if (withoutput)
    assert( model->getNumberOfOutputs() == dataset.getOutputs(0).size() );
  // azzera le variabili
  hits = 0;
  missed = 0;
  accuracy = 0.0;
  error = 0.0;
  // per ogni elemento del dataset
  for (uint elem = 0; elem < dataset.getSize(); ++elem) {
    // imposta l'input nel modello
    model->setInputs(dataset[elem].input);
    // avvia il calcolo
    model->compute();
    if (withoutput) {
      // controlla la risposta e l'errore restituiti dal modello
      checkModelResponse(elem) ? ++hits : ++missed;
      error += lastModelError(elem);
    }
    // salva l'output del modello
    saveLastOutputs(dataset[elem].id);
  } // end for elem
  if (withoutput) {
    // calcola i valori finali
    accuracy = (hits*100.0) / dataset.getSize();
    error = error / dataset.getSize();
  }
  return;
} // End method start

// ===============
// PRIVATE METHODS
// ===============

/**
 * Method checkModelResponse
 *
 * Controlla l'ultimo output del modello e lo confronta con l'i-esimo output
 * del dataset. Restituisce true se la risposta del modello e` uguale a quella
 * del dataset rispetto alla soglia impostata (entrambi maggiori o entrambi
 * minori).
 */
bool Tester::checkModelResponse(uint i) const {
  const real TH = threshold;
  for (uint k = 0; k < dataset[i].output.size(); ++k) {
    if ( ((dataset[i].output[k] > TH) && (model->getOutput(k) <= TH)) ||
         ((dataset[i].output[k] <= TH) && (model->getOutput(k) > TH)) )
      return false;
  } // end for k
  return true;
} // End method chekModelResponse

/**
 * Method lastModelError
 *
 * Restituisce l'errore quadratico dell'ultimo output del modello rispetto all'
 * output dell'i-esimo elemento nel dataset.
 * Viene restituito l'errore secondo la formula:
 *   E = (1/2) * Sum( (d(j)-y(j))^2 )
 */
real Tester::lastModelError(uint i) const {
  real err = 0.0;
  for (uint k = 0; k < dataset[i].output.size(); ++k)
    err += pow(dataset[i].output[k] - model->getOutput(k), 2);
  return err / 2;
} // End method lastModelError

/**
 * Method saveLastOutputs
 *
 * Appende nel file "resfile" gli ultimi output restituiti dal modello
 * preceduti dall'id (una stringa qualunque) passato come parametro:
 *   id, last_output[1], ..., last_output[n]
 * Se resfile non contiene un nome di file esce senza compiere nulla.
 */
void Tester::saveLastOutputs(const std::string& id) const {
  if (resfile.empty()) return;
  std::ofstream ofs;
  // modifica il formato di stampa per i numeri floating point
  std::streamsize prprec = ofs.precision(5);
  std::ios::fmtflags prflag =
      ofs.setf(std::ios::scientific, std::ios::floatfield);
  // apre il file e aggiunge una riga con i risultati
  ofs.open(resfile.c_str(),  std::ios::out | std::ios::app);
  if (!ofs.is_open()) throw file_error("In Trainer::saveLastOutputs");
  ofs <<id;
  for (uint i = 0; i < model->getNumberOfOutputs(); ++i)
    ofs <<"," <<model->getOutput(i);
  ofs <<std::endl;
  ofs.close();
  // ripristina il formato stampa
  ofs.precision(prprec);
  ofs.setf(prflag, std::ios::floatfield);
  return;
} // End method saveLastOutputs
