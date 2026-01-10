这里放置了三个版本的cuda 加速llama2， 实际上，run_naive.cu, run_version1.cu "减速" llama2推理

根据cpu version的火焰图， matmul占据了大部分的时间。引入cuda加速矩阵乘法后，产生大量HOST, DEVICE内存交换。它们成为了新的性能瓶颈。在cuda_naive_result.ipynb, cuda_verson1_result.ipynb中可见， 内存交换的开销占据了总开销的9成以上

详细的运行结果与性能对比在三个ipynb文件，最后的结果是llama2推理速度加速了7倍左右

实验配置： colab Nvidia GPU Tesla T4