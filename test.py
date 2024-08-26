#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 12:44:30 2024

@author: kalen
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms, models
from PIL import Image
from typing import Optional, Tuple, Type
import math

# 一 图像编码
from ViT_Encoder import ImageReader,PatchEmbed, ImageEncoderViT
# 1 简单的编码测试
# (1) 读取数据测试
image_reader = ImageReader(size=(256, 256)) # 初始化定义将读取的图片转为shape = [256, 256]
image = '/Users/kalen/Desktop/Python_env/my_paper/cat.jpg'
output_from_image_reader = image_reader(image)
print(output_from_image_reader.shape) # shape = [1, 3, 256, 256]

# (2) 图片编码测试（已经将ImageEmbed类打包进了EImageEncoderViT里）
image_encoder = ImageEncoderViT()
output_from_untrained_ImageEncoderViT = image_encoder(output_from_image_reader)
print(output_from_untrained_ImageEncoderViT.shape) # shape = [1, 256, 256, 16]

# 2 读取chackpoint测试
# (1) 读取checkpoint
checkpoint = '/Users/kalen/Desktop/Python_env/my_paper/sam_vit_b_01ec64.pth'
checkpoint = torch.load(checkpoint)

# (2) 实例化model，并读取checkpoint
model = ImageEncoderViT()
model.load_state_dict(checkpoint)

# (3) 测试图像数据
output_from_checkpoint_ImageEncoderViT = model(output_from_image_reader)
print(output_from_checkpoint_ImageEncoderViT.shape) # shape = [1, 256, 256, 16]

# 二 自然语言编码
from transformers import BertTokenizer, BertModel
# 工作流程: 自然语言描述 -> BertTokenizer -> encoded tensor -> BertModel -> output
# 1 定义自然语言描述
text = 'I want to work hard to get the CVPR 2025'

# 2 加载tokenizer和BERT模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# 3 用tokenizer和bert对自然语言编码
output_from_tokenizer = tokenizer(text, return_tensors="pt") #  return_tensors="pt"参数表示希望输出的是torch.Tensor
print("output_from_tokenizer的输出为:", output_from_tokenizer) # 但实际输出的是一个3个元素的字典，形如{'input_ids: tensor, 'token_type_ids': tensor, 'attention_mask': tensor}

output_from_bert_model = bert_model(**output_from_tokenizer) # 由于output_from_tokenizer是个字典，所以前面要加两个星号
'''bert_model(**output_from_tokenizer)的写法，等效于model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], token_type_ids=inputs.get("token_type_ids"))'''

# output_from_bert_model其实也是个字典，即{'last_hidden_state': tensor, 'pooler_output': tensor}
print("output_from_bert_model的last_hidden_state输出为:", output_from_bert_model.last_hidden_state.shape)
print("output_from_bert_model的poller_output输出为:", output_from_bert_model.pooler_output.shape)





