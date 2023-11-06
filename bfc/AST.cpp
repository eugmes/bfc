#include "AST.h"

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
  void dump(OpAST *op, llvm::raw_ostream &os);
  void dump(OpASTList *opList, llvm::raw_ostream &os);
  void dump(ModuleAST *node, llvm::raw_ostream &os);
  void dump(ModIndexOpAST *node, llvm::raw_ostream &os);
  void dump(ModDataOpAST *node, llvm::raw_ostream &os);
  void dump(InputOpAST *node, llvm::raw_ostream &os);
  void dump(OutputOpAST *node, llvm::raw_ostream &os);
  void dump(LoopOpAST *node, llvm::raw_ostream &os);

  // Actually print spaces matching the current indentation level
  void indent(llvm::raw_ostream &os) {
    for (int i = 0; i < curIndent; i++)
      os << " ";
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
#define INDENT(os)                                                             \
  Indent level_(curIndent);                                                    \
  indent(os);

// Dispatch to a generic operations to the appropriate subclass using RTTI
void ASTDumper::dump(OpAST *op, llvm::raw_ostream &os) {
  llvm::TypeSwitch<OpAST *>(op)
      .Case<ModIndexOpAST, ModDataOpAST, InputOpAST, OutputOpAST, LoopOpAST>(
          [&](auto *node) { this->dump(node, os); })
      .Default([&](OpAST *) {
        // No match, fallback to a generic message
        INDENT(os);
        os << "<unknown Op, kind " << op->getKind() << ">\n";
      });
}

/// A "block", or a list of operations
void ASTDumper::dump(OpASTList *opList, llvm::raw_ostream &os) {
  INDENT(os);
  os << "Block {\n";
  for (auto &op : *opList)
    dump(op.get(), os);
  indent(os);
  os << "} // Block\n";
}

void ASTDumper::dump(ModIndexOpAST *node, llvm::raw_ostream &os) {
  INDENT(os);
  os << "ModIndex " << node->getValue() << " " << loc(node) << "\n";
}

void ASTDumper::dump(ModDataOpAST *node, llvm::raw_ostream &os) {
  INDENT(os);
  os << "ModData " << node->getValue() << " " << loc(node) << "\n";
}

void ASTDumper::dump(InputOpAST *node, llvm::raw_ostream &os) {
  INDENT(os);
  os << "Input " << loc(node) << "\n";
}

void ASTDumper::dump(OutputOpAST *node, llvm::raw_ostream &os) {
  INDENT(os);
  os << "Output " << loc(node) << "\n";
}

void ASTDumper::dump(LoopOpAST *node, llvm::raw_ostream &os) {
  INDENT(os);
  os << "Loop " << loc(node) << "\n";
  dump(node->getBody(), os);
}

void ASTDumper::dump(ModuleAST *node, llvm::raw_ostream &os) {
  INDENT(os);
  os << "Module:\n";
  dump(node->getBody(), os);
}

namespace bf {

void dump(ModuleAST &module, llvm::raw_ostream &os) {
  ASTDumper().dump(&module, os);
}

} // namespace bf
