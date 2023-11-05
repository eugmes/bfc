#ifndef BF_PARSER_H
#define BF_PARSER_H

#include "AST.h"
#include "Lexer.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <optional>
#include <utility>
#include <vector>

namespace bf {

class Parser {
public:
  /// Create a Parser for the supplied lexer.
  Parser(Lexer &lexer) : lexer(lexer) {}

  std::unique_ptr<ModuleAST> parseModule() {
    lexer.getNextToken(); // prime the lexer

    OpASTList ops;
    while (auto op = parseOp()) {
      ops.push_back(std::move(op));
    }

    // If we didn't reach EOF, there was an error during parsing
    if (lexer.getCurToken() != tok_eof)
      return parseError<ModuleAST>("nothing", "at end of module");

    return std::make_unique<ModuleAST>(
        std::make_unique<OpASTList>(std::move(ops)));
  }

private:
  Lexer &lexer;

  std::unique_ptr<OpAST> parseOp() {
    auto token = lexer.getCurToken();
    auto location = lexer.getLastLocation();

    std::unique_ptr<OpAST> result;

    switch (token) {
    case tok_eof:
      return nullptr;

    case tok_gt:
      result = std::make_unique<ModPtrOpAST>(location, 1);
      break;
    case tok_lt:
      result = std::make_unique<ModPtrOpAST>(location, -1);
      break;
    case tok_plus:
      result = std::make_unique<ModValOpAST>(location, 1);
      break;
    case tok_minus:
      result = std::make_unique<ModValOpAST>(location, -1);
      break;
    case tok_period:
      result = std::make_unique<OutputOpAST>(location);
      break;
    case tok_comma:
      result = std::make_unique<InputOpAST>(location);
      break;
    case tok_open_paren:
      return parseLoopOp();
    case tok_close_paren:
      return nullptr;
    }

    lexer.getNextToken();
    return result;
  }

  std::unique_ptr<LoopOpAST> parseLoopOp() {
    auto location = lexer.getLastLocation();
    lexer.consume(tok_open_paren);
    OpASTList ops;

    while (auto op = parseOp()) {
      ops.push_back(std::move(op));
    }

    if (lexer.getCurToken() == tok_eof)
      return parseError<LoopOpAST>("]", "inside a loop");

    lexer.consume(tok_close_paren);
    return std::make_unique<LoopOpAST>(
        location, std::make_unique<OpASTList>(std::move(ops)));
  }

  /// Helper function to signal errors while parsing, it takes an argument
  /// indicating the expected token and another argument giving more context.
  /// Location is retrieved from the lexer to enrich the error message.
  template <typename R, typename T, typename U = const char *>
  std::unique_ptr<R> parseError(T &&expected, U &&context = "") {
    auto curToken = lexer.getCurToken();
    llvm::errs() << "Parse error (" << lexer.getLastLocation().line << ", "
                 << lexer.getLastLocation().col << "): expected '" << expected
                 << "' " << context << " but has Token " << curToken;
    if (isprint(curToken))
      llvm::errs() << " '" << (char)curToken << "'";
    llvm::errs() << "\n";
    return nullptr;
  }
};

} // namespace bf

#endif // BF_PARSER_H
