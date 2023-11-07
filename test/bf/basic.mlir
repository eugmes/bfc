// RUN: bf-opt %s | bf-opt | FileCheck %s
module {
  bf.program(%index: !bf.data_index, %data: !bf.data_store) {
    bf.mod_data %data[%index] by 1
    %loop_result = bf.loop(%index1 = %index, %data) -> !bf.data_index {
      %index2 = bf.mod_index %index1 by 1
      bf.input %data[%index2]
      %index3 = bf.mod_index %index2 by -1
      bf.mod_data %data[%index3] by -1
      bf.yield %index3
    }
    %index4 = bf.mod_index %loop_result by 1
    bf.output %data[%index4]
  }
}
// CHECK: bf.program(%arg0: !bf.data_index, %arg1: !bf.data_store) {
// CHECK-NEXT: bf.mod_data %arg1[%arg0] by 1
// CHECK-NEXT: %0 = bf.loop(%arg2 = %arg0, %arg1) -> !bf.data_index {
// CHECK-NEXT: %2 = bf.mod_index %arg2 by 1
// CHECK-NEXT: bf.input %arg1[%2]
// CHECK-NEXT: %3 = bf.mod_index %2 by -1
// CHECK-NEXT: bf.mod_data %arg1[%3] by -1
// CHECK-NEXT: bf.yield %3
// CHECK-NEXT: }
// CHECK-NEXT: %1 = bf.mod_index %0 by 1
// CHECK-NEXT: bf.output %arg1[%1]
// CHECK-NEXT: }
