#include "examples/common.h"
#include <Python.h>
#include <boost/python.hpp>
#include <boost/python/def.hpp>
#include <boost/python/list.hpp>
#include <functional>
#include <string>

#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

// Modules
#include <tokens.hpp>

// Utility libraries
#include <filesystem>

#include <llama.h>

#include <iostream>

namespace py = boost::python;

llama_context* ctx = nullptr;

void load_model(const std::string& path) {
    gpt_params params;
    params.model = path.c_str();
    params.n_ctx = 2048;

    if (!std::filesystem::exists(path)) {
        std::string error = "Failed to load model at \"";
        error += path;
        error += "\". File not found";
        PyErr_SetString(PyExc_FileNotFoundError, error.c_str());
        py::throw_error_already_set();
    }

    ctx = llama_init_from_gpt_params(params);

    if (!ctx) {
        std::string error = "Failed to load LLaMA model (I don't have more info)";
        PyErr_SetString(PyExc_ImportError, error.c_str());
        py::throw_error_already_set();
    }


    params.embedding = true;
}

void check_ctx_status() {
    if (!ctx) {
        std::string error = "LLaMA model not loaded. Call load_model before using any functionality!";
        PyErr_SetString(PyExc_RuntimeError, error.c_str());
        py::throw_error_already_set();
    }
}

// Function wrappers (can't use lambdas or bind sorry)
std::vector<pyllama::Token> tokenize_wrapper(char* const str) {
    check_ctx_status();
    return pyllama::tokenize(ctx, str);
}


BOOST_PYTHON_MODULE(pyllamacpp) {
    using namespace py;
    // Define iterators
    class_<std::vector<pyllama::Token>>("TokenList")
        .def(vector_indexing_suite<std::vector<pyllama::Token>>());

    def("load_model", load_model);
    def("tokenize", tokenize_wrapper);

    class_<pyllama::Token>("Token", no_init)
        .def_readwrite("token", &pyllama::Token::token)
        .def_readwrite("representation", &pyllama::Token::representation);
}
