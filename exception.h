#ifndef EXCEPTION_H_
#define EXCEPTION_H_

#include <stdexcept>
#include <string>

/**
 * Class file_error
 *
 * Eccezione per errori generici su file (il file non esiste, non si e`
 * riusciti ad aprirlo, ecc.)
 */
class file_error : public std::exception
{
  public:
    explicit file_error ( const std::string& what_arg ) throw() :
        what_arg(what_arg) { }
    virtual ~file_error() throw() { }
    virtual const char* what() const throw() {
      return what_arg.c_str();
    }
  private:
    std::string what_arg;
}; // End class file_error

/**
 * Class read_error
 *
 * Eccezione per errori generici nella lettura di inputs (formato non valido o
 * non riconosciuto, fuori dal range aspettato, ecc.)
 */
class read_error : public std::exception
{
  public:
    explicit read_error ( const std::string& what_arg ) throw() :
        what_arg(what_arg) { }
    virtual ~read_error() throw() { }
    virtual const char* what() const throw() {
      return what_arg.c_str();
    }
  private:
    std::string what_arg;
}; // End class read_error

#endif /* EXCEPTION_H_ */
