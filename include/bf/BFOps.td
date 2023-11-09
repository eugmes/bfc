#ifndef BF_OPS
#define BF_OPS

include "bf/BFTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def AllocOp : BF_Op<"alloc"> {
  let summary = "allocate the data store";

  let results = (outs Arg<DataStoreType, "data store", [MemAlloc]>:$data);

  let assemblyFormat = "`:` type($data) attr-dict";
}

def ModDataOp : BF_Op<"mod_data"> {
  let summary = "modify the data";

  let arguments = (ins Index:$index, Arg<DataStoreType, "data store", [MemRead, MemWrite]>:$data, IndexAttr:$amount);

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

def SetDataOp : BF_Op<"set_data"> {
  let summary = "set the data to a constant value";

  let arguments = (ins Index:$index, Arg<DataStoreType, "data store", [MemWrite]>:$data, IndexAttr:$value);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins "Value":$index, "Value":$data, "int64_t":$value)>];

  let assemblyFormat = "$data `[` $index `]` `to` $value attr-dict";

  let extraClassDeclaration = [{
    void setValue(int64_t value) {
      setValueAttr(IntegerAttr::get(IndexType::get(getContext()), value));
    }
  }];
}

def OutputOp : BF_Op<"output"> {
  let summary = "output the data";

  let arguments = (ins Index:$index, DataStoreType:$data);

  let assemblyFormat = "$data `[` $index `]` attr-dict";
}

def InputOp : BF_Op<"input"> {
  let summary = "input the data";

  let arguments = (ins Index:$index, DataStoreType:$data);

  let assemblyFormat = "$data `[` $index `]` attr-dict";
}

def LoopOp : BF_Op<"loop", [RecursiveMemoryEffects,
                            SingleBlock,
                            AllTypesMatch<["index", "result"]>]> {
  let summary = "loop";

  let arguments = (ins Index:$index, Arg<DataStoreType, "data store", [MemRead]>:$data);
  let results = (outs Index:$result);
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

def WhileOp : BF_Op<"while", [RecursiveMemoryEffects, SingleBlock, NoRegionArguments, NoTerminator]> {
  let summary = "simplified loop";

  let arguments = (ins Index:$index, Arg<DataStoreType, "data store", [MemRead]>:$data);
  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins "Value":$index, "Value":$data)>];

  let hasCanonicalizer = 1;

  let assemblyFormat = "$data `[` $index `]` $region attr-dict";
}

def YieldOp : BF_Op<"yield", [HasParent<"LoopOp">, Pure, Terminator]> {
  let summary = "yield result from a loop body";

  let arguments = (ins Index:$index);

  let assemblyFormat = "operands attr-dict";
}

def ProgramOp : BF_Op<"program", [IsolatedFromAbove, AutomaticAllocationScope, SingleBlock, NoRegionArguments, NoTerminator]> {
  let summary = "program operation";

  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins)>];

  let assemblyFormat = "$region attr-dict";
}

#endif // BF_OPS
