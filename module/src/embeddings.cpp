#include <algorithm>
#include <embeddings.hpp>
#include <iterator>
#include "tokens.hpp"
#include <common.h>
#include <llama.h>

// Base implementation
std::optional<std::vector<float>> generate_from_llama_tokens(const std::vector<llama_token>& llama_tokens, llama_context* ctx, size_t n_threads) {
    // Run inference
    if (::llama_eval(ctx, llama_tokens.data(), llama_tokens.size(), 0, n_threads) > 0) {
        return std::nullopt;
    }

    std::vector<float> embeddings(llama_n_embd(ctx));
    float* llama_embeddings = ::llama_get_embeddings(ctx);
    std::copy(llama_embeddings, llama_embeddings + embeddings.size(), embeddings.begin());

    return embeddings;
}

std::optional<std::vector<float>> pyllama::generate_embeddings(const std::string &str, llama_context* ctx, size_t n_threads) {
    // Add space to get the same behaviour as the OG tokenizer
    std::string fixed_string = " " + str;
    // Generate tokens from string
    auto tokens = llama_tokenize(ctx, fixed_string, true);

    return generate_from_llama_tokens(tokens, ctx, n_threads);
}

std::optional<std::vector<float>> pyllama::generate_embeddings_from_tokens(const std::vector<Token> &tokens, llama_context* ctx, size_t n_threads) {
    // Convert back to llama_tokens
    std::vector<llama_token> llama_tokens;
    std::transform(tokens.begin(), tokens.end(), std::back_inserter(llama_tokens), [](const Token& tok) {
        return tok.token;
    });
    return generate_from_llama_tokens(llama_tokens, ctx, n_threads);
}
