// RUN: bf-opt %s | bf-opt | FileCheck %s
module {
  bf.program(%index: !bf.data_index, %data: !bf.data_store) {
    %data1 = bf.mod_data %data[%index] by 1
    %loop_result:2 = bf.loop(%index1 = %index, %data2 = %data1) -> (!bf.data_index, !bf.data_store) {
      %index2 = bf.mod_ptr %index1 by 1
      %data3 = bf.input %data2[%index2]
      %index3 = bf.mod_ptr %index2 by -1
      %data4 = bf.mod_data %data3[%index3] by -1
      bf.yield %index3, %data4
    }
    %index4 = bf.mod_ptr %loop_result#0 by 1
    bf.output %loop_result#1[%index4]
  }
}
// CHECK: bf.program(%arg0: !bf.data_index, %arg1: !bf.data_store) {
// CHECK-NEXT: %0 = bf.mod_data %arg1[%arg0] by 1
// CHECK-NEXT: %index_result, %data_result = bf.loop(%arg2 = %arg0, %arg3 = %0) -> (!bf.data_index, !bf.data_store) {
// CHECK-NEXT: %2 = bf.mod_ptr %arg2 by 1
// CHECK-NEXT: %3 = bf.input %arg3[%2]
// CHECK-NEXT: %4 = bf.mod_ptr %2 by -1
// CHECK-NEXT: %5 = bf.mod_data %3[%4] by -1
// CHECK-NEXT: bf.yield %4, %5
// CHECK-NEXT: }
// CHECK-NEXT: %1 = bf.mod_ptr %index_result by 1
// CHECK-NEXT: bf.output %data_result[%1]
// CHECK-NEXT: }
