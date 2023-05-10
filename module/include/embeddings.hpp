#pragma once

#include <vector>
#include <string>

#include <tokens.hpp>

#include "llama.h"

namespace pyllama {
    std::vector<float> generate_emebddings(const std::string& str);
    std::vector<float> generate_embeddings_from_tokens(const std::vector<Token>& tokens);
}
