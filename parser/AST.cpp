#include "bf/AST.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

using namespace bf;

namespace {

// RAII helper to manage increasing/decreasing the indentation as we traverse
// the AST
struct Indent {
  Indent(int &level) : level(level) { ++level; }
  ~Indent() { --level; }
  int &level;
};

/// Helper class that implement the AST tree traversal and print the nodes along
/// the way. The only data member is the current indentation level.
class ASTDumper {
public:
  void dump(OpAST *op);
  void dump(OpASTList *opList);
  void dump(ModuleAST *node);
  void dump(ModPtrOpAST *node);
  void dump(ModValOpAST *node);
  void dump(InputOpAST *node);
  void dump(OutputOpAST *node);
  void dump(LoopOpAST *node);

  // Actually print spaces matching the current indentation level
  void indent() {
    for (int i = 0; i < curIndent; i++)
      llvm::errs() << " ";
  }
  int curIndent = 0;
};

} // namespace

/// Return a formatted string for the location of any node
template <typename T> static std::string loc(T *node) {
  const auto &loc = node->loc();
  return (llvm::Twine("@") + *loc.file + ":" + llvm::Twine(loc.line) + ":" +
          llvm::Twine(loc.col))
      .str();
}

// Helper Macro to bump the indentation level and print the leading spaces for
// the current indentations
#define INDENT()                                                               \
  Indent level_(curIndent);                                                    \
  indent();

// Dispatch to a generic operations to the appropriate subclass using RTTI
void ASTDumper::dump(OpAST *op) {
  llvm::TypeSwitch<OpAST *>(op)
      .Case<ModPtrOpAST, ModValOpAST, InputOpAST, OutputOpAST, LoopOpAST>(
          [&](auto *node) { this->dump(node); })
      .Default([&](OpAST *) {
        // No match, fallback to a generic message
        INDENT();
        llvm::errs() << "<unknown Op, kind " << op->getKind() << ">\n";
      });
}

/// A "block", or a list of operations
void ASTDumper::dump(OpASTList *opList) {
  INDENT();
  llvm::errs() << "Block {\n";
  for (auto &op : *opList)
    dump(op.get());
  indent();
  llvm::errs() << "} // Block\n";
}

void ASTDumper::dump(ModPtrOpAST *node) {
  INDENT();
  llvm::errs() << "ModPtr " << node->getValue();
  llvm::errs() << " " << loc(node) << "\n";
}

void ASTDumper::dump(ModValOpAST *node) {
  INDENT();
  llvm::errs() << "ModVal " << node->getValue();
  llvm::errs() << " " << loc(node) << "\n";
}

void ASTDumper::dump(InputOpAST *node) {
  INDENT();
  llvm::errs() << "Input " << loc(node) << "\n";
}

void ASTDumper::dump(OutputOpAST *node) {
  INDENT();
  llvm::errs() << "Output " << loc(node) << "\n";
}

void ASTDumper::dump(LoopOpAST *node) {
  INDENT();
  llvm::errs() << "Loop " << loc(node) << "\n";
  dump(node->getBody());
}

void ASTDumper::dump(ModuleAST *node) {
  INDENT();
  llvm::errs() << "Module:\n";
  dump(node->getBody());
}

namespace bf {

void dump(ModuleAST &module) { ASTDumper().dump(&module); }

} // namespace bf
