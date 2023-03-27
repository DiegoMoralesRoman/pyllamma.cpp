#include <Python.h>
#include <boost/python.hpp>
#include <boost/python/list.hpp>
#include <iostream>

#include "ggml.h"
#include "llama.h"
#include "pyerrors.h"
#include "utils.h"

llama_context * ctx;
size_t num_threads = 11;

namespace py = boost::python;

llama_token SPACE_TOKEN;
llama_token START_TOKEN = 1;

void load_model(char const* filename) {
    ggml_time_init();
    auto lparams = llama_context_default_params();

    lparams.seed = time(NULL);
    lparams.embedding = true;

    ctx = llama_init_from_file(filename, lparams);

    SPACE_TOKEN = llama_tokenize(ctx, " ", false)[0];

    if (ctx == NULL)
        PyErr_SetString(PyExc_FileNotFoundError, "Failed to load model");
}

py::list internal_generate_embeddings(std::vector<llama_token>& tokens) {
    //
    if (tokens.size() > 0) {
        if (llama_eval(ctx, tokens.data(), tokens.size(), 0, num_threads)) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to make inference");
            return py::list{};
        }
    }
    const auto calculated_embd = llama_get_embeddings_vector(ctx);
    py::list ret_list;
    for (auto component : calculated_embd)
        ret_list.append(component);
    return ret_list;
}

py::list calculate_embeddings(char const* prompt) {
    std::string updated_prompt(prompt);
    updated_prompt += ' ';

    std::vector<llama_token> embd;

    embd = ::llama_tokenize(ctx, updated_prompt.c_str(), true);
    return internal_generate_embeddings(embd);
}


struct Token {
    llama_token token;
    const char* representation;

    Token(llama_token token, const char* representation)
        : token(token), representation(representation) {}

    Token() 
        : token(0), representation("") {}
};

py::list from_tokens(py::list list) {
    size_t len = py::len(list);

    std::vector<llama_token> tokens{START_TOKEN};

    for (size_t i = 0; i < len; i++) {
        py::object item = list[i];
        auto extracted = py::extract<Token>(item);
        if (extracted.check())
            tokens.push_back(extracted().token);
        else {
            std::string error_string = "List element at position "
                + std::to_string(i)
                + " is not of type Token";
            PyErr_SetString(PyExc_TypeError, error_string.c_str());
            return py::list{};
        }
    }

    tokens.push_back(SPACE_TOKEN);

    return internal_generate_embeddings(tokens);
}

py::list tokenize(char const* prompt) {
    std::vector<llama_token> tokens;
    tokens = llama_tokenize(ctx, prompt, false);
    py::list ret_list;
    for (auto& token : tokens)
        ret_list.append(Token(token, llama_token_to_str(ctx, token)));


    return ret_list;
}

BOOST_PYTHON_MODULE(pyllamacpp) {
    using namespace py;
    def("load_model", load_model);
    def("generate_embeddings", calculate_embeddings);
    def("generate_from_tokens", from_tokens);
    class_<Token>("Token", init<>())
      .def_readonly("token", &Token::token)
      .def_readonly("representation", &Token::representation);

    def("tokenize", tokenize);
}

