// RUN: bf-opt %s -canonicalize | FileCheck %s
module {
  bf.program(%arg0: !bf.data_index, %arg1: !bf.data_store) {
    // CHECK: bf.mod_data %arg1[%arg0] by 2
    bf.mod_data %arg1[%arg0] by 1
    bf.mod_data %arg1[%arg0] by 1
    // CHECK-NEXT: bf.output %arg1[%arg0]
    bf.output %arg1[%arg0]
    // CHECK-NEXT: %0 = bf.mod_index %arg0 by 1
    %0 = bf.mod_index %arg0 by 1
    // CHECK-NEXT: bf.mod_data %arg1[%arg0] by -2
    bf.mod_data %arg1[%arg0] by -2
    // CHECK-NEXT: bf.mod_data %arg1[%0] by 2
    bf.mod_data %arg1[%0] by 2
  }
}
