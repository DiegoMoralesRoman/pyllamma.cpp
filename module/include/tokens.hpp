#pragma once

#include <vector>
#include <string>

#include <llama.h>

namespace pyllama {
    struct Token {
        llama_token token;
        const char* representation;
        bool operator==(const Token& other) const;
    };

std::vector<Token> tokenize(llama_context* ctx, const std::string& str);
}

