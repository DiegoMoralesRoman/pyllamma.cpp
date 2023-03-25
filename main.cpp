#include "utils.h"
#include "ggml.h"
#include "llama.h"

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

std::vector<double> softmax(const std::vector<float>& logits) {
    std::vector<double> probs(logits.size());
    float max_logit = logits[0];
    for (float v : logits) max_logit = std::max(max_logit, v);
    double sum_exp = 0.0;
    for (size_t i = 0; i < logits.size(); i++) {
        // Subtract the maximum logit value from the current logit value for numerical stability
        float logit = logits[i] - max_logit;
        double exp_logit = std::exp(logit);
        sum_exp += exp_logit;
        probs[i] = exp_logit;
    }
    for (size_t i = 0; i < probs.size(); i++) probs[i] /= sum_exp;
    return probs;
}


int main() {
    // has to be called once at the start of the program to init ggml stuff
    ggml_time_init();

    gpt_params params;
    params.model = "models/7B/ggml-model-q4_0.bin";

    // Model has max of 2048 tokens

    params.seed = time(NULL);
    params.n_predict = 128;
    params.embedding = true;

    llama_context * ctx;

    // load the model
    {
        auto lparams = llama_context_default_params();

        lparams.n_ctx      = params.n_ctx;
        lparams.n_parts    = params.n_parts;
        lparams.seed       = params.seed;
        lparams.f16_kv     = params.memory_f16;
        lparams.logits_all = params.perplexity;
        lparams.use_mlock  = params.use_mlock;
        lparams.embedding  = params.embedding;

        ctx = llama_init_from_file(params.model.c_str(), lparams);

        if (ctx == NULL) {
            fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, params.model.c_str());
            return 1;
        }
    }
    fprintf(stderr, "sampling parameters: temp = %f, top_k = %d, top_p = %f, repeat_last_n = %i, repeat_penalty = %f\n", params.temp, params.top_k, params.top_p, params.repeat_last_n, params.repeat_penalty);

    int n_past = 0;

    params.prompt = "Esto es una prueba de funcionamiento";
    // Add a space in front of the first character to match OG llama tokenizer behavior
    params.prompt.insert(0, 1, ' ');

    // tokenize the prompt
    // auto embd_inp = ::llama_tokenize(ctx, params.prompt, true);

    const int n_ctx = llama_n_ctx(ctx);

    // params.n_predict = std::min(params.n_predict, n_ctx - (int) embd_inp.size());

    std::vector<llama_token> embd;

    int last_n_size = params.repeat_last_n;
    std::vector<llama_token> last_n_tokens(last_n_size);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);

    std::vector<std::pair<std::string, std::vector<float>>> embeddings;

    for (std::string& str : std::vector<std::string>{"Ordenadores", "La literatura del siglo XIX", "La historia de la computación", "Procesadores", "Matemáticas"}) {
        std::cout << "Tokenizing " << str << '\n';
        embd = ::llama_tokenize(ctx, str, true);
        std::cout << "Calculating embeddings...\n";
        if (embd.size() > 0) {
            if (llama_eval(ctx, embd.data(), embd.size(), n_past, params.n_threads)) {
                fprintf(stderr, "%s : failed to eval\n", __func__);
                return 1;
            }
        }

        std::cout << "Getting embeddings\n";
        const auto calculated_embd = llama_get_embeddings_vector(ctx);
        embeddings.push_back({str, calculated_embd});
    }

    
    auto cosine_similarity = [](const std::vector<float>& v1, const std::vector<float>& v2) -> float {
        if (v1.size() != v2.size()) {
            throw std::invalid_argument("Vectors must have the same length.");
        }

        float dot_product = 0.0f;
        float v1_magnitude = 0.0f;
        float v2_magnitude = 0.0f;

        for (size_t i = 0; i < v1.size(); i++) {
            dot_product += v1[i] * v2[i];
            v1_magnitude += v1[i] * v1[i];
            v2_magnitude += v2[i] * v2[i];
        }

        v1_magnitude = std::sqrt(v1_magnitude);
        v2_magnitude = std::sqrt(v2_magnitude);

        if (v1_magnitude == 0.0f || v2_magnitude == 0.0f) {
            throw std::runtime_error("Vectors must not have zero magnitude.");
        }

        return dot_product / (v1_magnitude * v2_magnitude);
    };

    auto base = embeddings[0];
    for (auto e : embeddings) {
        std::cout << "Cosine similarity between " << base.first << " and " << e.first << ": " <<
            cosine_similarity(base.second, e.second) << '\n';
    }

    

    llama_free(ctx);

    return 0;
}
