#ifndef BF_OPS
#define BF_OPS

include "bf/BFTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def ModIndexOp : BF_Op<"mod_index", [Pure, AllTypesMatch<["index", "result"]>]> {
  let summary = "modify the data pointer";

  let arguments = (ins DataIndexType:$index, IndexAttr:$amount);
  let results = (outs DataIndexType:$result);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins "Value":$index, "int64_t":$amount)>];

  let hasCanonicalizer = 1;

  let assemblyFormat = "$index `by` $amount attr-dict";

  let extraClassDeclaration = [{
    void setAmount(int64_t value) {
      setAmountAttr(IntegerAttr::get(IndexType::get(getContext()), value));
    }
  }];
}

def ModDataOp : BF_Op<"mod_data", []> {
  let summary = "modify the data";

  let arguments = (ins DataIndexType:$index, Arg<DataStoreType, "data store", [MemRead, MemWrite]>:$data, IndexAttr:$amount);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins "Value":$index, "Value":$data, "int64_t":$amount)>];

  let hasCanonicalizer = 1;

  let assemblyFormat = "$data `[` $index `]` `by` $amount attr-dict";

  let extraClassDeclaration = [{
    void setAmount(int64_t value) {
      setAmountAttr(IntegerAttr::get(IndexType::get(getContext()), value));
    }
  }];
}

def SetDataOp : BF_Op<"set_data", []> {
  let summary = "set the data to a constant value";

  let arguments = (ins DataIndexType:$index, Arg<DataStoreType, "data store", [MemWrite]>:$data, IndexAttr:$value);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins "Value":$index, "Value":$data, "int64_t":$value)>];

  let assemblyFormat = "$data `[` $index `]` `to` $value attr-dict";

  let extraClassDeclaration = [{
    void setValue(int64_t value) {
      setValueAttr(IntegerAttr::get(IndexType::get(getContext()), value));
    }
  }];
}

def OutputOp : BF_Op<"output", []> {
  let summary = "output the data";

  let arguments = (ins DataIndexType:$index, DataStoreType:$data);

  let assemblyFormat = "$data `[` $index `]` attr-dict";
}

def InputOp : BF_Op<"input", []> {
  let summary = "input the data";

  let arguments = (ins DataIndexType:$index, DataStoreType:$data);

  let assemblyFormat = "$data `[` $index `]` attr-dict";
}

def LoopOp : BF_Op<"loop", [RecursiveMemoryEffects,
                            SingleBlock,
                            AllTypesMatch<["index", "result"]>]> {
  let summary = "loop";

  let arguments = (ins DataIndexType:$index, Arg<DataStoreType, "data store", [MemRead]>:$data);
  let results = (outs DataIndexType:$result);
  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins "Value":$index, "Value":$data)>];

  let hasCustomAssemblyFormat = 1;
  let hasRegionVerifier = 1;
  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    BlockArgument getIndexArgument() {
      return getBody()->getArgument(0);
    }
  }];
}

def YieldOp : BF_Op<"yield", [HasParent<"LoopOp">, Pure, Terminator]> {
  let summary = "yield result from a loop body";

  let arguments = (ins DataIndexType:$index);

  let assemblyFormat = "operands attr-dict";
}

def ProgramOp : BF_Op<"program", [IsolatedFromAbove, AutomaticAllocationScope, SingleBlock, NoTerminator]> {
  let summary = "program operation";

  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;
  // TODO: add argument names
  let builders = [OpBuilder<(ins)>];

  let hasCustomAssemblyFormat = 1;
  let hasRegionVerifier = 1;

  let extraClassDeclaration = [{
    BlockArgument getIndexArgument() {
      return getBody()->getArgument(0);
    }

    BlockArgument getDataArgument() {
      return getBody()->getArgument(1);
    }
  }];
}

#endif // BF_OPS
