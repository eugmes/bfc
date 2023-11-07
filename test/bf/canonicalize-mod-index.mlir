// RUN: bf-opt %s -canonicalize | FileCheck %s
module {
  bf.program(%arg0: !bf.data_index, %arg1: !bf.data_store) {
    // CHECK: bf.mod_index %arg0 by 3
    %0 = bf.mod_index %arg0 by 1
    %1 = bf.mod_index %0 by 1
    %2 = bf.mod_index %1 by 1
    bf.output %arg1[%2]
    // CHECK: bf.mod_index %arg0 by 2
    %3 = bf.mod_index %2 by -1
    bf.output %arg1[%3]
    %4 = bf.mod_index %3 by -2
    // CHECK: bf.output %arg1[%arg0]
    bf.output %arg1[%4]
  }
}
