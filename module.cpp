#include <Python.h>
#include <boost/python.hpp>
#include <iostream>

#include "ggml.h"
#include "llama.h"
#include "pyerrors.h"
#include "utils.h"

llama_context * ctx;

void load_model(char const* filename) {
    ggml_time_init();
    auto lparams = llama_context_default_params();

    lparams.seed = time(NULL);
    lparams.embedding = true;

    ctx = llama_init_from_file(filename, lparams);

    if (ctx == NULL)
        PyErr_SetString(PyExc_FileNotFoundError, "Failed to load model");
}


boost::python::list calculate_embeddings(char const* prompt) {
    gpt_params params;
    params.prompt = prompt;
    // Add a space in front of the first character to match OG llama tokenizer behavior
    params.prompt.insert(0, 1, ' ');
    std::vector<llama_token> embd;

    embd = ::llama_tokenize(ctx, prompt, true);
    if (embd.size() > 0) {
        if (llama_eval(ctx, embd.data(), embd.size(), 0, params.n_threads)) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to make inference");
        }
    }
    const auto calculated_embd = llama_get_embeddings_vector(ctx);
    boost::python::list ret_list;
    for (auto component : calculated_embd)
        ret_list.append(component);
    return ret_list;
}

BOOST_PYTHON_MODULE(pyllamacpp) {
    using namespace boost::python;
    def("load_model", load_model);
    def("generate_embeddings", calculate_embeddings);
}
