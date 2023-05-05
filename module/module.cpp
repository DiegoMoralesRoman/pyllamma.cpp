#include <Python.h>
#include <boost/python.hpp>
#include <boost/python/list.hpp>

namespace py = boost::python;

#include <iostream>
void foo() {
    std::cout << "Hola, esto es una prueba funcionando desde C++\n";
}

BOOST_PYTHON_MODULE(pyllamacpp) {
    using namespace py;
    def("say_hello", foo);
}
