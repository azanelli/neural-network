#include "global.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cassert>
#include "exception.h"

typedef Global::uint uint;
typedef Global::real real;

// ======================
// PRIVATE STATIC MEMBERS
// ======================

uint Global::rseed;
std::map<std::string, std::string> Global::parameters;

// =====================
// PUBLIC STATIC METHODS
// =====================

/**
 * Method readParameters
 *
 * Legge i parametri passati nell'array di stringhe di caratteri argv (di
 * dimensione argc) che sono nella forma:
 *   --param_name param_value
 * oppure
 *   -param_name param_value
 * Se nel nome del parametro ci sono trattini (-) iniziali o finali questi
 * vengono eliminati. Se un parametro non ha valore (un flag) allora il
 * suo valore viene impostato con il nome stesso del parametro (una stringa
 * non vuota).
 * I parametri letti possono essere reperiti successivamente con il metodo
 * getParam.
 */
void Global::readParameters(int argc, char** argv) {
  if (argc == 0)
    return;
  int index;
  for (index = 0; index < argc; ++index) {
    std::string strarg(argv[index]);
    if (strarg[0] == '-' && !isNumeric(strarg)) {
      trim(strarg, "-");
      if ( (index+1) < argc ) {
        std::string opt(argv[index+1]);
        if ( (opt[0] == '-') && (!isNumeric(opt)) ) {
          parameters.insert( std::make_pair(strarg, strarg) );
        } else {
          parameters.insert( std::make_pair(strarg, opt) );
          ++index;
        }
      } else {
        parameters.insert( std::make_pair(strarg, strarg) );
      } // end if ( (index+1) < argc )
    } // end if (strarg[0] == '-' && !isNumeric(strarg))
  } // end for index
  return;
} // End method readParameters

/**
 * Method getNumberOfParams
 *
 * Restituisce il numero di parametri attualmente memorizzati (accessibili con
 * i metodi getParam).
 */
uint Global::getNumberOfParams() {
  return parameters.size();
} // End method getNumberOfParams

/**
 * Method getParamValue
 *
 * Restituisce il valore del parametro con indice i. Non e` garantito che
 * l'indice di un parametro rimanga costante.
 * Il tempo di accesso e` lineare al numero di parametri.
 */
const std::string& Global::getParamValue(uint i) {
  if (i >= parameters.size())
    throw std::out_of_range("In method Global::getParamValue");
  std::map<std::string,std::string>::iterator it = parameters.begin();
  for (uint k = 0; k < i; ++k, ++it) { }
  return (*it).second;
} // End method getParamValue

/**
 * Method getParamKey
 *
 * Restituisce il nome del parametro con indice i. Non e` garantito che
 * l'indice di un parametro rimanga costante.
 * Il tempo di accesso e` lineare al numero di parametri.
 */
const std::string& Global::getParamKey(uint i) {
  if (i >= parameters.size())
    throw std::out_of_range("In method Global::getParamValue");
  std::map<std::string,std::string>::iterator it = parameters.begin();
  for (uint k = 0; k < i; ++k, ++it) { }
  return (*it).first;
} // End method getParamKey

/**
 * Method getParam
 *
 * Restituisce il valore (una stringa) del parametro con nome name. Per
 * acquisire i parametri vedere il metodo readParameters, per inserirne di
 * nuovi vedere il metodo setParam.
 * Se il parametro richiesto non esiste il metodo restituisce una stringa
 * vuota.
 */
const std::string& Global::getParam(const std::string& name) {
  return parameters[name];
} // End method getParam

/**
 * Method setParam
 *
 * Imposta nel parametro di nome name (se non esiste lo crea) il valore value.
 * Restituisce il valore value.
 */
const std::string& Global::setParam(const std::string& name,
    const std::string& value) {
  parameters[name] = value;
  return value;
} // End method setParam

/**
 * Function toString
 *
 * Converte un numero reale in una stringa.
 */
std::string Global::toString(real value) {
  char buffer[32];
  sprintf(buffer, "%g", value);
  return std::string(buffer);
} // End method toString

/**
 * Function toString
 *
 * Converte un intero in una stringa.
 */
std::string Global::toString(int value) {
  char buffer[32];
  sprintf(buffer, "%d", value);
  return std::string(buffer);
} // End method toString

/**
 * Function toString
 *
 * Converte un intero positivo (tipo uint) in una stringa.
 */
std::string Global::toString(uint value) {
  char buffer[32];
  sprintf(buffer, "%d", value);
  return std::string(buffer);
} // End method toString

/**
 * Function toReal
 *
 * Converte una stringa nel corrispondente numero reale (tipo real). Se la
 * conversione non e` possibile restituisce il numero 0.
 */
Global::real Global::toReal(const std::string& value) {
  return strtod(value.c_str(),NULL);
} // End method toReal

/**
 * Function toFloat
 *
 * Converte una stringa nel corrispondente numero float. Se la conversione non
 * e` possibile restituisce il numero 0.
 */
float Global::toFloat(const std::string& value) {
  return atof(value.c_str());
} // End method toFloat

/**
 * Function toInt
 *
 * Converte una stringa nel corrispondente numero intero (tipo int). Se la
 * conversione non e` possibile restituisce il numero 0.
 */
int Global::toInt(const std::string& value) {
  return atoi(value.c_str());
} // End method toInt

/**
 * Function toUint
 *
 * Converte una stringa nel corrispondente numero intero senza segno (tipo
 * uint). Se la conversione non e` possibile restituisce il numero 0.
 */
Global::uint Global::toUint(const std::string& value) {
  return atoi(value.c_str());
} // End method toUint

/**
 * Function isNumeric
 *
 * Restituisce true se la stringa passata rappresenta un numero.
 */
bool Global::isNumeric (const std::string& value) {
  if (value.empty())
    return false;
  std::istringstream iss(value);
  double number = 0.0;
  if (iss >> number)
    return true;
  return false;
} // End method isNumeric

/**
 * Function setRandSeed
 *
 * Inizializza il generatore di numeri casuali con il seme passato.
 */
void Global::setRandSeed(uint seed) {
  rseed = seed;
  srand(rseed);
} // End method setRandSeed

/**
 * Function getRandSeed
 *
 * Restituisce il seme iniziale utilizzato per generare numeri casuali.
 */
uint Global::getRandSeed() {
  return rseed;
} // End method getRandSeed

/**
 * Function getRand
 *
 * Restituisce un numero casuale (intero) nell'intervallo [start, end].
 */
int Global::getRand(uint start, uint end) {
  assert(start <= end);
  return ( rand()%(end-start+1) ) + start;
} // End method getRand

/**
 * Function trim
 *
 * Data una stringa str, la funzione elimina tutti i caratteri contenuti nella
 * stringa di caratteri t dall'inizio e dalla fine della stringa str finche`
 * non trova un carattere differente. La funzione restituisce la stringa str
 * stessa (dopo il "trattamento").
 */
const std::string& Global::trim(std::string& str, const char* t) {
  std::string::size_type startpos = str.find_first_not_of(t);
  std::string::size_type endpos = str.find_last_not_of(t);
  if ( (startpos == std::string::npos) || (endpos == std::string::npos) )
    str = "";
  else
    str = str.substr(startpos, endpos-startpos+1);
  return str;
} // End method trim

/**
 * Function split
 *
 * Data una stringa str, la funzione split separa la stringa str in
 * sottostringhe prendendo come carattere delimitatore delim.
 * La funzione restituisce un puntatore ad un vettore contenente tutte le
 * sottostringhe. E` compito dell'utente eliminare il vettore (l'oggetto puntato
 * dal puntatore) dopo averlo utilizzato.
 */
std::vector<std::string>* Global::split(const std::string& str, char delim) {
  std::vector<std::string>* vector = new std::vector<std::string>();
  std::stringstream stream(str);
  std::string token;
  while (std::getline(stream, token, delim)) vector->push_back(token);
  return vector;
} // End method split
