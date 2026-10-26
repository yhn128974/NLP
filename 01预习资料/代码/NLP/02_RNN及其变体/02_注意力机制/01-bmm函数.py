import torch

"""
    bmm和matmul对比总结
        1- 都能够实现矩阵乘法运算。matmul是串行计算，效率低；bmm是并行计算，效率高
        2- 如果matmul的张量形状不等，会进行广播机制
        3- bmm的要求如下：
            3.1- 参与运算的两个张量的形状有且只能是三维。形状的含义(batch_size,行数,列数)
            3.2- 参与运算的两个张量的batch_size必须相等
            3.3- 参与运算的两个张量中，前一个张量的列数 与 后一个张量的行数 必须相同
            
    后面在计算注意力的时候，如果满足bmm的要求，推荐使用bmm；当然可以无脑使用matmul，因为兼容性更好
"""
if __name__ == '__main__':
    # 准备数据
    mat_1 = torch.randint(low=1,high=5,size=(10,3,4))
    # mat_1 = torch.randint(low=1,high=5,size=(1,3,4))
    mat_2 = torch.randint(low=1,high=5,size=(10,4,5))

    # matmul
    matmul_result = mat_1 @ mat_2
    print("matmul_result：",matmul_result.shape)

    # bmm
    bmm_result = torch.bmm(mat_1,mat_2)
    print("bmm_result：", bmm_result.shape)
