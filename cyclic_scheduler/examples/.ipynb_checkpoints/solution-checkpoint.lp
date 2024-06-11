\ Model model
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
 
Subject To
 dependency_a_b: - throughput_0 - a + b >= 0
 dependency_a_b: - throughput_1 - a + b >= 0
 dependency_a_b: - throughput_0 - a + b >= 0
 dependency_a_b: - throughput_1 - a + b >= 0
 dependency_a_b: - throughput_0 - a + b >= 0
 dependency_a_b: - throughput_1 - a + b >= 0
 dependency_a_b: - throughput_0 - a + b >= 0
 dependency_a_b: - throughput_1 - a + b >= 0
 dependency_a_b: - throughput_0 - a + b >= 0
 dependency_a_b: - throughput_1 - a + b >= 0
 dependency_a_b: - throughput_0 - a + b >= 0
 dependency_a_b: - throughput_1 - a + b >= 0
 dependency_b_c: - throughput_0 - b + c >= 0
 dependency_b_c: - throughput_1 - b + c >= 0
 dependency_b_c: - throughput_0 - b + c >= 0
 dependency_b_c: - throughput_1 - b + c >= 0
 memory_allocation_0: ring_buffer_0_0 + ring_buffer_1_0 <= 10
 (d+0)/6: - 0.1666666666666667 ring_buffer_0_0 + C12 <= 0
 (d+0)/1: - ring_buffer_0_1 + C13 <= 0
 (d+1)/6: - 0.1666666666666667 ring_buffer_0_0 + C14 <= 0.1666666666666667
 (d+0)/1: - ring_buffer_0_1 + C15 <= 0
 (d+2)/6: - 0.1666666666666667 ring_buffer_0_0 + C16 <= 0.3333333333333333
 (d+0)/1: - ring_buffer_0_1 + C17 <= 0
 (d+3)/6: - 0.1666666666666667 ring_buffer_0_0 + C18 <= 0.5
 (d+0)/1: - ring_buffer_0_1 + C19 <= 0
 (d+4)/6: - 0.1666666666666667 ring_buffer_0_0 + C20 <= 0.6666666666666666
 (d+0)/1: - ring_buffer_0_1 + C21 <= 0
 (d+5)/6: - 0.1666666666666667 ring_buffer_0_0 + C22 <= 0.8333333333333333
 (d+0)/1: - ring_buffer_0_1 + C23 <= 0
 R29: - throughput_0 + a - b + C20 >= 0
 R30: - throughput_1 + a - b + C23 >= 0
 R31: - throughput_0 + a - b + C16 >= 0
 R32: - throughput_1 + a - b + C23 >= 0
 R33: - throughput_0 + a - b + C12 >= 0
 R34: - throughput_1 + a - b + C23 >= 0
 R35: - throughput_0 + a - b + C14 >= -1
 R36: - throughput_1 + a - b + C23 >= 0
 R37: - throughput_0 + a - b + C22 >= 0
 R38: - throughput_1 + a - b + C23 >= 0
 R39: - throughput_0 + a - b + C18 >= 0
 R40: - throughput_1 + a - b + C23 >= 0
 (d+0)/2: - 0.5 ring_buffer_1_0 + C24 <= 0
 (d+0)/1: - ring_buffer_1_1 + C25 <= 0
 (d+1)/2: - 0.5 ring_buffer_1_0 + C26 <= 0.5
 (d+0)/1: - ring_buffer_1_1 + C27 <= 0
 R45: - throughput_0 + b - c + C26 >= 0
 R46: - throughput_1 + b - c + C27 >= 0
 R47: - throughput_0 + b - c + C24 >= 0
 R48: - throughput_1 + b - c + C27 >= 0
 R49: - throughput_0 + a - a + C28 >= 0
 R50: - throughput_0 - a + a - C28 >= -1
 R51: - throughput_0 + a - a + C29 >= 0
 R52: - throughput_0 - a + a - C29 >= -1
 R53: - throughput_0 + a - c + C30 >= 0
 R54: - throughput_0 - a + c - C30 >= -1
 R55: - throughput_0 + a - a + C31 >= 0
 R56: - throughput_0 - a + a - C31 >= -1
 R57: - throughput_0 + a - c + C32 >= 0
 R58: - throughput_0 - a + c - C32 >= -1
 R59: - throughput_0 + a - c + C33 >= 0
 R60: - throughput_0 - a + c - C33 >= -1
 R61: 4 throughput_0 <= 1
 R62: - throughput_0 + b - b + C34 >= 0
 R63: - throughput_0 - b + b - C34 >= -1
 R64: 2 throughput_0 <= 1
Bounds
 C28 free
 C29 free
 C30 free
 C31 free
 C32 free
 C33 free
 C34 free
Generals
 C12 C13 C14 C15 C16 C17 C18 C19 C20 C21 C22 C23 C24 C25 C26 C27 C28 C29
 C30 C31 C32 C33 C34
End
