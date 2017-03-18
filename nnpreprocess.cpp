#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <vector>
#include <string>
#include <cmath>
#include "global.h"
#include "exception.h"

typedef Global::uint uint;

/**
 * Function printHelp
 *
 * Stampa su standard output l'help del programma.
 */
void printHelp() {
  std::cout <<"argv[1] : nome del file" <<std::endl;
  std::cout <<"argv[2] : numero totale di partizioni" <<std::endl;
  std::cout <<"argv[3] : partizione di test in [0,n-1]" <<std::endl;
  std::cout <<"argv[4] : seme per numeri casuali" <<std::endl;
  return;
} // End function printHelp

/**
 * Functions begin
 *
 * Restituisce l'indice del primo elemento della k-esima partizione:
 *   - tot : totale elementi
 *   - n   : numero di partizioni
 *   - k   : partizione desiderata
 */
uint begin(uint tot, uint n, uint k) {
  uint r = tot%n;
  if (k <= r) return k * ( floor(tot/double(n)) + 1 );
  return (k * floor(tot/double(n)) ) + r;
} // End function begin

/**
 * Function end
 *
 * Restituisce l'indice dell'ultimo elemento della k-esima partizione.
 *   - tot : totale elementi
 *   - n   : numero di partizioni
 *   - k   : partizione desiderata
 */
uint end(uint tot, uint n, uint k) {
  if (k == (n-1)) return tot;
  return begin(tot,n,k+1);
} // End function end

/**
 * Function main
 *
 * Divide il file passato in due file .ts e .tr, il primo che rappresenta la
 * partizione richieste, il secondo l'unione delle altre partizioni.
 * Mantenendo lo stesso seme casuale si ottengono sempre le stesse partizioni
 * (percui si puo` ruotare il .ts).
 */
int main(int argc, char **argv) {
  if (argc == 1) {
    printHelp();
    return 0;
  }

  // Legge i parametri
  std::string filename(argv[1]);
  uint n = Global::toUint(std::string(argv[2]));
  uint k = Global::toUint(std::string(argv[3]));
  uint rseed = Global::toUint(std::string(argv[4]));

  // Inizializza il generatore di numeri casuali
  Global::setRandSeed(rseed);

  // Carica il file in memoria (riga per riga)
  std::vector<std::string*> filebuffer;
  std::string* line;
  std::ifstream infile(filename.c_str());
  if (!infile.is_open())
    throw file_error("In function main");
  while (infile.good()) {
    line = new std::string();
    std::getline(infile, *line);
    filebuffer.push_back(line);
  } // end while (file.good())
  infile.close();

  // Crea un vettore di permutazione casuale
  std::vector<uint> p(filebuffer.size());
  for (uint i = 0; i < p.size(); ++i) p[i] = i;
  for (uint i = p.size(); i != 0; --i)
    std::swap( p[i-1], p[Global::getRand(0,i-1)] );

  // Crea gli indici per i due file
  std::vector<uint> trindex, tsindex;
  for (uint i = 0; i < p.size(); ++i) {
    if (i < begin(filebuffer.size(),n,k) || i >= end(filebuffer.size(),n,k))
      trindex.push_back(p[i]);
    else
      tsindex.push_back(p[i]);
  }

  // Ordina gli indici delle partizioni
  std::sort(trindex.begin(), trindex.end());
  std::sort(tsindex.begin(), tsindex.end());

  // Scrive la partizione tr su file
  std::string trname = filename + ".tr";
  std::ofstream trfile(trname.c_str());
  if (!trfile.is_open())
    throw file_error("In function main");
  for (uint i = 0; i < trindex.size(); ++i)
    trfile <<*(filebuffer[trindex[i]]) <<std::endl;
  trfile.close();

  // Scrive le partizione ts su file
  std::string tsname = filename + ".ts";
  std::ofstream tsfile(tsname.c_str());
  if (!tsfile.is_open())
    throw file_error("In function main");
  for (uint i = 0; i < tsindex.size(); ++i)
    tsfile <<*(filebuffer[tsindex[i]]) <<std::endl;
  tsfile.close();

  return 0;
} // End function main
