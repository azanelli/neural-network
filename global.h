#ifndef GLOBAL_H_
#define GLOBAL_H_

#include <string>
#include <vector>
#include <map>
#include <cstdlib>

/**
 * Class Global
 *
 * Contiene definizioni, funzioni e oggetti utilizzati globalmente all'interno
 * dell'applicazione.
 */
class Global {
  public:
    typedef unsigned int uint;
    typedef double real;

    static void readParameters ( int argc, char **argv );
    static uint getNumberOfParams ( );
    static const std::string& getParamValue ( uint i );
    static const std::string& getParamKey ( uint i );
    static const std::string& getParam ( const std::string& name );
    static const std::string& setParam ( const std::string& name,
        const std::string& value );
    static std::string toString ( real value );
    static std::string toString ( int value );
    static std::string toString ( uint value );
    static real toReal ( const std::string& value );
    static float toFloat ( const std::string& value );
    static int toInt ( const std::string& value );
    static uint toUint ( const std::string& value );
    static bool isNumeric ( const std::string& value );
    static void setRandSeed ( uint seed );
    static uint getRandSeed ( );
    static int getRand ( uint start = 0, uint end = RAND_MAX );
    static const std::string& trim ( std::string& str, const char* t = " ");
    static std::vector<std::string>* split ( const std::string& str,
        char delim = ' ' );

  private:
    static uint rseed;
    static std::map<std::string, std::string> parameters;

}; // End class Global

#endif /* GLOBAL_H_ */
