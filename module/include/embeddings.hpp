#pragma once

#include <vector>
#include <string>
#include <optional>

#include <tokens.hpp>

#include "llama.h"

constexpr size_t MAX_TOKENS = 2048;

namespace pyllama {
    std::optional<std::vector<float>> generate_embeddings(const std::string& str, llama_context* ctx, size_t n_threads);
    std::optional<std::vector<float>> generate_embeddings_from_tokens(const std::vector<Token>& tokens, llama_context* ctx, size_t n_threads);
}
