#!/usr/bin/env python3

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at:
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.


import os
import pickle
import argparse
import numpy as np
import tensorflow as tf
import neurite as ne
import voxelmorph as vxm

# reference
ref = (
    'If you find this script useful, please consider citing:\n\n'
    '\tM Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca\n'
    '\tSynthMorph: learning contrast-invariant registration without acquired images\n'
    '\tIEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022\n'
    '\thttps://doi.org/10.1109/TMI.2021.3116879\n'
)


# parse command line
bases = (argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter)
# ArgumentDefaultsHelpFormatter自动添加默认的值的信息到每一个帮助信息的参数中
# RawDescriptionHelpFormatter和RawTextHelpFormatter更好地控制文本描述的显示方式。
#   默认情况下，ArgumentParser对象在命令行帮助消息中对 description 和 epilog文本进行换行

# 初始化一个分析器
p = argparse.ArgumentParser(
    formatter_class=type('formatter', bases, {}),
    description=f'Train a SynthMorph model on images synthesized from label maps. {ref}',
)
# 在 add_argument 前，给属性名之前加上“- -”，就能将之变为可选参数。
# data organization parameters
p.add_argument('--label-dir', nargs='+', help='path or glob pattern pointing to input label maps')
# --label-dir : 指向输入标签映射的路径或全局模式
p.add_argument('--model-dir', default='models', help='model output directory')
# --model-dir : 模型输出目录
p.add_argument('--log-dir', help='optional TensorBoard log directory')
# --log-dir : 可选的 TensorBoard 日志目录
p.add_argument('--sub-dir', help='optional subfolder for logs and model saves')
# --sub-dir : 日志和模型保存的可选子文件夹

# generation parameters 生成参数
p.add_argument('--same-subj', action='store_true', help='generate image pairs from same label map')
# --same-subj: action='store_true': 当启动程序时必须添加并定义此参数时，用于指定功能 : （是否）从相同的标签映射生成图像对
p.add_argument('--blur-std', type=float, default=1, help='maximum blurring std. dev.')
# --blur-std: 最大模糊标准，如果不进行指定则默认为1
p.add_argument('--gamma', type=float, default=0.25, help='std. dev. of gamma')
# --gamma: gamma值的 标准差？
p.add_argument('--vel-std', type=float, default=0.5, help='std. dev. of SVF')
# --vel-std: 平滑矢量场(SVF)的 标准差？
p.add_argument('--vel-res', type=float, nargs='+', default=[16], help='SVF scale')
# --vel-res: SVF的尺度，参数可设置一个或多个，默认为一个16
p.add_argument('--bias-std', type=float, default=0.3, help='std. dev. of bias field')
# --bias-std: 偏置场的标准差
p.add_argument('--bias-res', type=float, nargs='+', default=[40], help='bias scale')
# --bias-res: 偏置场的尺度，参数可设置一个或多个，默认为一个40
p.add_argument('--out-shape', type=int, nargs='+', help='output shape to pad to''')
# --out-shape: 输出的形状被填补到输入参数大小，参数可设置一个或多个
p.add_argument('--out-labels', default='fs_labels.npy', help='labels to optimize, see README')
# --out-labels: 要优化的标签，请参阅自述文件，默认为fs_labels.npy

# training parameters 训练参数
p.add_argument('--gpu', type=str, default='0', help='ID of GPU to use')
# --gpu: 指定gpu的id，默认为0，即第一张卡
p.add_argument('--epochs', type=int, default=1500, help='training epochs')
# --epochs: 指定总的训练迭代数
p.add_argument('--batch-size', type=int, default=1, help='batch size')
# --batch-size: 指定批大小，默认为1
p.add_argument('--init-weights', help='optional weights file to initialize with')
#　--init-weight: 用于初始化的可选权重文件
p.add_argument('--save-freq', type=int, default=300, help='epochs between model saves')
# --save-freq: 模型保存的频率
p.add_argument('--reg-param', type=float, default=1., help='regularization weight')
# --reg-param: 正则化权重,默认为 1
p.add_argument('--lr', type=float, default=1e-4, help='learning rate')
# --lr: 学习率，默认为0.0001
p.add_argument('--init-epoch', type=int, default=0, help='initial epoch number')
# --init-epoch: 初始化代数？
p.add_argument('--verbose', type=int, default=1, help='0 silent, 1 bar, 2 line/epoch')
# --verbose: 冗余，默认为1，设置为0为silent，设置为1为bat，设置为2为行/代数？？？

# network architecture parameters 网络架构参数
p.add_argument('--int-steps', type=int, default=5, help='number of integration steps')
# --int-steps: 积分步骤数，默认为5？？？
p.add_argument('--enc', type=int, nargs='+', default=[64] * 4, help='U-Net encoder filters')
# --enc: U-Net编码器卷积数量，默认为4个64层
p.add_argument('--dec', type=int, nargs='+', default=[64] * 6, help='U-Net decorder filters')
# --dec: 解码器卷积数量，默认为6个64层

arg = p.parse_args()    # 属性给与args


# TensorFlow handling Tensorflow处理
device, nb_devices = vxm.tf.utils.setup_device(arg.gpu) # 返回设备 ID 和设备总数。

# Python assert（断言）用于判断一个表达式，在表达式条件为 false 的时候触发异常。
# 断言可以在条件不满足程序运行的情况下直接返回错误，而不必等待程序运行后出现崩溃的情况

# 当batch_size不是gpu数量的整数倍时，报错
assert np.mod(arg.batch_size, nb_devices) == 0, \
    f'batch size {arg.batch_size} not a multiple of the number of GPUs {nb_devices}'
# 当tensorflow不是2及以后的版本时，报错
assert tf.__version__.startswith('2'), f'TensorFlow version {tf.__version__} is not 2 or later'


# prepare directories   准备字典
sub_dir = "sub_dir"
model_dir = r"C:\Users\LeoHuang_redmiG\OneDrive\pythonProject\Liver_CT\preprocessed data\model"
model_dir = os.path.join(model_dir, sub_dir)
os.makedirs(model_dir, exist_ok=True)   # os.makedirs用来创建多层目录，只有在目录不存在时创建目录


# labels and label maps 标签和标签映射
# labels_in, label_maps = vxm.py.utils.load_labels(arg.label_dir) # 加载标签映射并返回唯一标签列表以及所有标签映射，后者为ndarray格式
labels_in, label_maps = vxm.py.utils.load_labels(r"C:\Users\LeoHuang_redmiG\OneDrive\pythonProject\Liver_CT\preprocessed data\downsampled")
# 生成器 gen
gen = vxm.generators.synthmorph(
    label_maps,
    batch_size=arg.batch_size,
    same_subj=arg.same_subj,
    flip=True,
)
in_shape = label_maps[0].shape
# 如果out_labels是.npy格式，则？
# 如果out_labels是.pickle格式，则？
if arg.out_labels.endswith('.npy'):
    labels_out = sorted(x for x in np.load(arg.out_labels) if x in labels_in)
elif arg.out_labels.endswith('.pickle'):
    with open(arg.out_labels, 'rb') as f:
        labels_out = {k: v for k, v in pickle.load(f).items() if k in labels_in}
else:
    labels_out = labels_in


# model configuration   模型配置：输入形状，输出形状，输入标签列表，各参数
gen_args = dict(
    in_shape=in_shape,
    out_shape=arg.out_shape,
    in_label_list=labels_in,
    out_label_list=labels_out,
    warp_std=arg.vel_std,
    warp_res=arg.vel_res,
    blur_std=arg.blur_std,
    bias_std=arg.bias_std,
    bias_res=arg.bias_res,
    gamma_std=arg.gamma,
)

# 配准参数设置：积分步骤数，输入分辨率，积分分辨率，平滑矢量场分辨率，nb_unet_features？
reg_args = dict(
    int_steps=arg.int_steps,
    int_resolution=2,
    svf_resolution=2,
    nb_unet_features=(arg.enc, arg.dec),
)

# build model 创建模型
strategy = 'MirroredStrategy' if nb_devices > 1 else 'get_strategy'
# 如果cuda设备数量大于1，使用MirroredStrategy策略，否则使用get_strategy策略
with getattr(tf.distribute, strategy)().scope():

    # generation 生成
    gen_model_1 = ne.models.labels_to_image(**gen_args, id=0)   # ？
    gen_model_2 = ne.models.labels_to_image(**gen_args, id=1)   # ？
    ima_1, map_1 = gen_model_1.outputs  # ？
    ima_2, map_2 = gen_model_2.outputs  # ？

    # registration  配准
    inputs = gen_model_1.inputs + gen_model_2.inputs    
    reg_args['inshape'] = ima_1.shape[1:-1]
    reg_args['input_model'] = tf.keras.Model(inputs, outputs=(ima_1, ima_2))
    model = vxm.networks.VxmDense(**reg_args)
    flow = model.references.pos_flow
    pred = vxm.layers.SpatialTransformer(interp_method='linear', name='pred')([map_1, flow])

    # losses and compilation    # 损失和编译
    const = tf.ones(shape=arg.batch_size // nb_devices) # 常亮
    model.add_loss(vxm.losses.Dice().loss(map_2, pred) + const) # 相似性损失，此处使用的为soft dice
    model.add_loss(vxm.losses.Grad('l2', loss_mult=arg.reg_param).loss(None, flow)) # 平滑度损失，此处使用的是gradient
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=arg.lr)) # 配置网络的优化器和学习率
    model.summary()     # 打印模型各层的参数状况


# callbacks 回调
steps_per_epoch = 100   # 每一代有多少步
model_dir = r"C:\Users\LeoHuang_redmiG\OneDrive\pythonProject\Liver_CT\preprocessed data\model"
save_name = os.path.join(model_dir, '{epoch:05d}.h5')   # 保存为model_dir下"epoch代数.h5"文件
# save_name = os.path.join(arg.model_dir, '{epoch:05d}.h5')   # 保存为model_dir下"epoch代数.h5"文件
save = tf.keras.callbacks.ModelCheckpoint(      # 在每个训练期（epoch）后保存模型
    save_name,
    save_freq=steps_per_epoch * arg.save_freq,
)
callbacks = [save]      # 将其保存为list

# if arg.log_dir:         # 如果定义了log_dir，则使用keras自带的回调保存日志
#     log = tf.keras.callbacks.TensorBoard(
#         log_dir=arg.log_dir,
#         write_graph=False,
#     )
#     callbacks.append(log)
log = tf.keras.callbacks.TensorBoard(
    log_dir=r"C:\Users\LeoHuang_redmiG\OneDrive\pythonProject\Liver_CT\preprocessed data\log_dir",
    write_graph=False,
)
callbacks.append(log)


# initialize and fit 初始化并训练
if arg.init_weights:    # 如果有初始化权重则载入
    model.load_weights(arg.init_weights)
model.save(save_name.format(epoch=arg.init_epoch))
model.fit(      # 开始训练
    gen,
    initial_epoch=arg.init_epoch,
    epochs=arg.epochs,
    callbacks=callbacks,
    steps_per_epoch=steps_per_epoch,
    verbose=arg.verbose,
)
print(f'\nThank you for using SynthMorph! {ref}')