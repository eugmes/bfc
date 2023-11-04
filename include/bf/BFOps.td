#ifndef BF_OPS
#define BF_OPS

include "bf/BFTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def ModPtrOp : BF_Op<"mod_ptr", [Pure, AllTypesMatch<["input", "output"]>]> {
  let summary = "modify the data pointer";

  let arguments = (ins DataIndexType:$input, IndexAttr:$amount);
  let results = (outs DataIndexType:$output);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins "Value":$input, "int64_t":$amount)>];

  let assemblyFormat = "$input `by` $amount attr-dict";
}

def ModDataOp : BF_Op<"mod_data", [Pure, AllTypesMatch<["data", "output"]>]> {
  let summary = "modify the data";

  let arguments = (ins DataIndexType:$index, DataStoreType:$data, IndexAttr:$amount);
  let results = (outs DataStoreType:$output);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins "Value":$index, "Value":$data, "int64_t":$amount)>];

  let assemblyFormat = "$data `[` $index `]` `by` $amount attr-dict";
}

def OutputOp : BF_Op<"output", []> {
  let summary = "output the data";

  let arguments = (ins DataIndexType:$index, DataStoreType:$data);

  let assemblyFormat = "$data `[` $index `]` attr-dict";
}

def InputOp : BF_Op<"input", [AllTypesMatch<["data", "output"]>]> {
  let summary = "input the data";

  let arguments = (ins DataIndexType:$index, DataStoreType:$data);
  let results = (outs DataStoreType:$output);

  let assemblyFormat = "$data `[` $index `]` attr-dict";
}

def LoopOp : BF_Op<"loop", [RecursiveMemoryEffects,
                            SingleBlock,
                            AllTypesMatch<["index", "output_index"]>,
                            AllTypesMatch<["data", "output_data"]>]> {
  let summary = "loop";

  let arguments = (ins DataIndexType:$index, DataStoreType:$data);
  let results = (outs DataIndexType:$output_index, DataStoreType:$output_data);
  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins "Value":$index, "Value":$data)>];

  let hasCustomAssemblyFormat = 1;
}
// bf.loop(%ptr = <value>, %data = <value>) {
//   yield %ptr, %data
// }

def YieldOp : BF_Op<"yield", [HasParent<"LoopOp">, Pure, Terminator]> {
  let summary = "yield result from a loop body";

  let arguments = (ins DataIndexType:$index, DataStoreType:$data);

  let assemblyFormat = "attr-dict $data `,` $index";
}

def ProgramOp : BF_Op<"program", [IsolatedFromAbove, AutomaticAllocationScope, SingleBlock, NoTerminator]> {
  let summary = "program operation";

  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;
  // TODO: add argument names
  let builders = [OpBuilder<(ins)>];

  let hasCustomAssemblyFormat = 1;
}

// bf.program(%ptr, %data) {}

#endif // BF_OPS
