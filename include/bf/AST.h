#ifndef BF_AST_H
#define BF_AST_H

#include "bf/Lexer.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include <optional>
#include <utility>
#include <vector>

namespace bf {

/// Base class for all operation nodes.
class OpAST {
public:
  enum OpASTKind {
    Op_ModPtr,
    Op_ModVal,
    Op_Input,
    Op_Output,
    Op_Loop,
  };

  OpAST(OpASTKind kind, Location location) : kind(kind), location(location) {}
  virtual ~OpAST() = default;

  OpASTKind getKind() const { return kind; }

  const Location &loc() const { return location; }

private:
  const OpASTKind kind;
  Location location;
};

using OpASTList = std::vector<std::unique_ptr<OpAST>>;

class ModPtrOpAST : public OpAST {
  std::int64_t value;

public:
  ModPtrOpAST(Location location, std::int64_t value)
      : OpAST(Op_ModPtr, location), value(value) {}

  std::int64_t getValue() const { return value; }

  // LLVM style RTTI
  static bool classof(const OpAST *c) { return c->getKind() == Op_ModPtr; }
};

class ModValOpAST : public OpAST {
  std::int64_t value;

public:
  ModValOpAST(Location location, std::int64_t value)
      : OpAST(Op_ModVal, location), value(value) {}

  std::int64_t getValue() const { return value; }

  // LLVM style RTTI
  static bool classof(const OpAST *c) { return c->getKind() == Op_ModVal; }
};

class InputOpAST : public OpAST {
public:
  InputOpAST(Location location) : OpAST(Op_Input, location) {}

  // LLVM style RTTI
  static bool classof(const OpAST *c) { return c->getKind() == Op_Input; }
};

class OutputOpAST : public OpAST {
public:
  OutputOpAST(Location location) : OpAST(Op_Output, location) {}

  // LLVM style RTTI
  static bool classof(const OpAST *c) { return c->getKind() == Op_Output; }
};

class LoopOpAST : public OpAST {
  std::unique_ptr<OpASTList> body;

public:
  LoopOpAST(Location location, std::unique_ptr<OpASTList> body)
      : OpAST(Op_Loop, location), body(std::move(body)) {}

  OpASTList *getBody() { return body.get(); }

  // LLVM style RTTI
  static bool classof(const OpAST *c) { return c->getKind() == Op_Loop; }
};

class ModuleAST {
  std::unique_ptr<OpASTList> body;

public:
  ModuleAST(std::unique_ptr<OpASTList> body) : body(std::move(body)) {}

  OpASTList *getBody() { return body.get(); }
};

void dump(ModuleAST &);

} // namespace bf

#endif // BF_AST_H
