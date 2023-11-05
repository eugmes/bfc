// RUN: bf-opt %s | bf-opt | FileCheck %s

module {
    // CHECK-LABEL: bf.program
    bf.program(%index: !bf.data_index, %data: !bf.data_store) {
        // CHECK-LABEL: bf.loop
        bf.loop(%index1 = %index, %data1 = %data) -> (!bf.data_index, !bf.data_store) {
            // CHECK-LABEL: bf.yield
            bf.yield %index1, %data1
        }
    }
}
