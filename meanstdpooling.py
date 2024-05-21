class MeanStdPooling(nn.Module):
    def __init__(self):
        super(MeanStdPooling, self).__init__()


    def forward(self, x):
        batch_size, C, H, W = x.shape
        # print('x.shape', x.shape)

        # 使用全局平均池化层
        gap_layer = nn.AdaptiveAvgPool2d((1, 1))
        mean_pool_map_4x4 = []
        var_pool_map_4x4 = []
        mean_pool_map_2x2 = []
        var_pool_map_2x2 = []
        mean_pool_map_1x1 = []
        var_pool_map_1x1 = []

        # 计算需要填充的高度和宽度
        pad_height = (4 - (H % 4)) % 4
        pad_width = (4 - (W % 4)) % 4

        x_4x4 = F.pad(x, (0, pad_width, 0, pad_height))  # (left, right, top, bottom)

        _, _, H_padded, W_padded = x_4x4.shape

        # 将图像分割成16块 (4x4)
        blocks_4x4 = [
            x_4x4[:, :, i * H_padded // 4: (i + 1) * H_padded // 4, j * W_padded // 4: (j + 1) * W_padded // 4]
            for i in range(4) for j in range(4)]

        for block in blocks_4x4:
            # 对输入张量应用全局平均池化
            mean_pool = gap_layer(block)
            # print('mean_pool_4x4.shape', mean_pool.shape)
            mean_pool_map_4x4.append(mean_pool)
            # 对输入张量应用方差池化
            mean = torch.mean(block, dim=(2, 3), keepdim=True)
            var_pool = torch.mean((block - mean) ** 2, dim=(2, 3), keepdim=True)
            # print('var_pool_4x4.shape', var_pool.shape)
            var_pool_map_4x4.append(var_pool)

        # 将mean_pool_map转换为张量
        mean_pool_tensor_4x4 = torch.cat(mean_pool_map_4x4, dim=3)  # 按最后一维拼接
        # print('mean_pool_tensor_4x4.shape', mean_pool_tensor_4x4.shape)
        mean_pool_tensor_4x4 = mean_pool_tensor_4x4.view(batch_size, C, 4, 4)
        # print('mean_pool_tensor_4x4.shape', mean_pool_tensor_4x4.shape)
        var_pool_tensor_4x4 = torch.cat(var_pool_map_4x4, dim=3)  # 按最后一维拼接
        # print('var_pool_tensor_4x4.shape', var_pool_tensor_4x4.shape)
        var_pool_tensor_4x4 = var_pool_tensor_4x4.view(batch_size, C, 4, 4)
        # print('var_pool_tensor_4x4.shape', var_pool_tensor_4x4.shape)

        # 计算需要填充的高度和宽度
        pad_height = (H % 2)  # 如果高度是奇数，则pad_height为1；否则为0
        pad_width = (W % 2)  # 如果宽度是奇数，则pad_width为1；否则为0

        x_2x2 = F.pad(x, (0, pad_width, 0, pad_height))  # (left, right, top, bottom)

        _, _, H_padded, W_padded = x_2x2.shape

        # 将图像分割成4块 (2x2)
        blocks_2x2 = [
            x_2x2[:, :, i * H_padded // 2: (i + 1) * H_padded // 2, j * W_padded // 2: (j + 1) * W_padded // 2]
            for i in range(2) for j in range(2)]

        for block in blocks_2x2:
            # 对输入张量应用全局平均池化
            mean_pool = gap_layer(block)
            # print('mean_pool_2x2.shape', mean_pool.shape)
            mean_pool_map_2x2.append(mean_pool)
            # 对输入张量应用方差池化
            mean = torch.mean(block, dim=(2, 3), keepdim=True)
            var_pool = torch.mean((block - mean) ** 2, dim=(2, 3), keepdim=True)
            # print('var_pool_2x2.shape', var_pool.shape)
            var_pool_map_2x2.append(var_pool)

        # 将mean_pool_map转换为张量
        mean_pool_tensor_2x2 = torch.cat(mean_pool_map_2x2, dim=3)  # 按最后一维拼接
        # print('mean_pool_tensor_2x2.shape', mean_pool_tensor_2x2.shape)
        mean_pool_tensor_2x2 = mean_pool_tensor_2x2.view(batch_size, C, 2, 2)
        # print('mean_pool_tensor_2x2.shape', mean_pool_tensor_2x2.shape)
        var_pool_tensor_2x2 = torch.cat(var_pool_map_2x2, dim=3)  # 按最后一维拼接
        # print('var_pool_tensor_2x2.shape', var_pool_tensor_2x2.shape)
        var_pool_tensor_2x2 = var_pool_tensor_2x2.view(batch_size, C, 2, 2)
        # print('var_pool_tensor_2x2.shape', var_pool_tensor_2x2.shape)

        # 对输入张量应用全局平均池化
        mean_pool_map_1x1 = gap_layer(x)
        # print('mean_pool_map_1x1.shape', mean_pool_map_1x1.shape)
        # 对输入张量应用方差池化
        mean = torch.mean(x, dim=(2, 3), keepdim=True)
        var_pool_map_1x1 = torch.mean((x - mean) ** 2, dim=(2, 3), keepdim=True)
        mean_pool_tensor_1x1 = mean_pool_map_1x1.view(batch_size, C, 1, 1)
        var_pool_tensor_1x1 = var_pool_map_1x1.view(batch_size, C, 1, 1)

        # print('-------------------------------------------------------------------')

        # 将每个张量展平成一维张量
        mean_pool_tensor_4x4 = mean_pool_tensor_4x4.view(batch_size, C, -1)  # 形状变为 (batchsize, channel, 16)
        mean_pool_tensor_2x2 = mean_pool_tensor_2x2.view(batch_size, C, -1)  # 形状变为 (batchsize, channel, 4)
        mean_pool_tensor_1x1 = mean_pool_tensor_1x1.view(batch_size, C, -1)  # 形状变为 (batchsize, channel, 1)
        var_pool_tensor_4x4 = var_pool_tensor_4x4.view(batch_size, C, -1)  # 形状变为 (batchsize, channel, 16)
        var_pool_tensor_2x2 = var_pool_tensor_2x2.view(batch_size, C, -1)  # 形状变为 (batchsize, channel, 4)
        var_pool_tensor_1x1 = var_pool_tensor_1x1.view(batch_size, C, -1)  # 形状变为 (batchsize, channel, 1)

        mean_feature_map = torch.cat((mean_pool_tensor_4x4, mean_pool_tensor_2x2, mean_pool_tensor_1x1),
                                     dim=2)  # 形状变为 (batchsize, channel, 21)
        # print('mean_feature_map.shape', mean_feature_map.shape)
        deviation_feature_map = torch.cat((var_pool_tensor_4x4, var_pool_tensor_2x2, var_pool_tensor_1x1),
                                          dim=2)  # 形状变为 (batchsize, channel, 21)
        # print('deviation_feature_map.shape', deviation_feature_map.shape)

        # print('-------------------------------------------------------------------')

        mean_feature_map_mean = torch.mean(mean_feature_map, dim=2, keepdim=True)
        mean_feature_vector = torch.mean((mean_feature_map - mean_feature_map_mean) ** 2, dim=2, keepdim=True)
        # print('mean_feature_vector.shape', mean_feature_vector.shape)
        deviation_feature_vector = torch.mean(deviation_feature_map, dim=2, keepdim=True)
        # print('deviation_feature_vector.shape', deviation_feature_vector.shape)

        global_feature = torch.cat((mean_feature_vector, deviation_feature_vector), dim=1).view(batch_size, 1, -1)
        # print('global_feature.shape', global_feature.shape)

        return global_feature
