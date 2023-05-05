#include <Python.h>
#include <boost/python.hpp>
#include <boost/python/list.hpp>
#include <string>

#include <iostream>

namespace py = boost::python;

void load_model(const std::string& path) {
    std::cout << "Loading model from path (C++) " << path << '\n';
}

BOOST_PYTHON_MODULE(pyllamacpp) {
    using namespace py;
    def("load_model", load_model);
}
