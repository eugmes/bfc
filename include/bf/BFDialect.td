#ifndef BF_DIALECT
#define BF_DIALECT

def BF_Dialect : Dialect {
  let name = "bf";
  let summary = "A BF MLIR dialect.";
  let cppNamespace = "::mlir::bf";

  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{
    void registerTypes();
  }];
}

class BF_Op<string mnemonic, list<Trait> traits =[]>
    : Op<BF_Dialect, mnemonic, traits>;

#endif // BF_DIALECT
