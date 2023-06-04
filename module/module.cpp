#include "embeddings.hpp"
#include "examples/common.h"
#include <Python.h>
#include <boost/python.hpp>
#include <boost/python/def.hpp>
#include <boost/python/list.hpp>
#include <boost/python/object.hpp>
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
    // Initialize parameters
    gpt_params params;
    params.model = path.c_str();
    params.n_ctx = 2048;
    params.embedding = true;

    params.seed = time(NULL);

    // Check if model exists
    if (!std::filesystem::exists(path)) {
        std::string error = "Failed to load model at \"";
        error += path;
        error += "\". File not found";
        PyErr_SetString(PyExc_FileNotFoundError, error.c_str());
        py::throw_error_already_set();
    }

    // Load model from the parameters
    ctx = llama_init_from_gpt_params(params);

    if (!ctx) {
        std::string error = "Failed to load LLaMA model (I don't have more info)";
        PyErr_SetString(PyExc_ImportError, error.c_str());
        py::throw_error_already_set();
    }
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

std::vector<float> generate_embeddings_wrapper(char* const str) {
    std::string prompt(str);
    auto inference = pyllama::generate_embeddings(prompt, ctx, std::thread::hardware_concurrency());
    if (inference.has_value()){
        return *inference;
    } else {
        // Throw exception
        std::string error = std::string("Failed to run inference on data:\n") + str;
        PyErr_SetString(PyExc_RuntimeError, error.c_str());
        py::throw_error_already_set();
    }
    return std::vector<float>(); // Default value (although it shouldn't be used)
}

std::vector<float> generate_embeddings_from_tokens_wrapper(py::list list) {
    // Convert list into std::vector
    size_t list_len = py::len(list);
    return std::vector<float>();
}


BOOST_PYTHON_MODULE(pyllamacpp) {
    using namespace py;
    // Define iterators
    class_<std::vector<pyllama::Token>>("TokenList")
        .def(vector_indexing_suite<std::vector<pyllama::Token>>());

    class_<std::vector<float>>("Embeddings")
        .def(vector_indexing_suite<std::vector<float>>());

    def("load_model", load_model);
    def("tokenize", tokenize_wrapper);

    def("generate_embeddings", generate_embeddings_wrapper);

    class_<pyllama::Token>("Token", no_init)
        .def_readwrite("token", &pyllama::Token::token)
        .def_readwrite("representation", &pyllama::Token::representation);
}
