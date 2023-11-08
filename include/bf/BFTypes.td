#ifndef BT_TYPES
#define BF_TYPES

include "mlir/IR/AttrTypeBase.td"
include "BF/BFDialect.td"

class BF_Type<string name, string typeMnemonic, list<Trait> traits =[]>
    : TypeDef<BF_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def DataStoreType : BF_Type<"DataStore", "data_store"> {
  let summary = "BF data store";
}

#endif // BF_TYPES
