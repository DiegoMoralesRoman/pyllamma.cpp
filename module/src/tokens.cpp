#include "common.h"
#include "llama.h"
#include <tokens.hpp>
#include <ranges>

using namespace pyllama;

bool Token::operator==(const Token &other) const {
    return token == other.token;
}

std::vector<Token> pyllama::tokenize(llama_context *ctx, const std::string& str) {
    auto tokens = llama_tokenize(ctx, str, false);

    // Create a transform_view with tokens as the input range
    auto transformed_tokens = tokens | std::views::transform([ctx](auto tok) -> Token {
        return Token {
            tok,
            llama_token_to_str(ctx, tok)
        };
    });

    // Convert the transform_view to a vector and return it
    return std::vector<Token>(transformed_tokens.begin(), transformed_tokens.end());
}
