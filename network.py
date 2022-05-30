"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

# Code Adapted from:
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
#
# BSD 3-Clause License
#
# Copyright (c) 2017,
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
# import network.mynn as  mynn
import torch.nn.functional as F

from Strawberry_Test.Utils.nms import nms
from boxes import AnchorsGenerator, get_topN_coords
from resnet import resnet50


from skimage import measure
from Strawberry_Test.hu_PartCrop import config as cfg


class CRM_Net(nn.Module):
    def __init__(self, num_classes, input_size):
        super(CRM_Net, self).__init__()
        self.backbone = resnet50(pretrained=True, num_classes=num_classes)
        self.num_classes = num_classes
        self.input_size_net = input_size

        self.num_features = 512 * self.backbone.expansion
        self.classifier = nn.Linear(self.num_features, num_classes)

        self.avg = nn.AdaptiveAvgPool2d(1)

        self.map_origin = nn.Conv2d(self.num_features, num_classes, 1, 1, 0)
        anchor_sizes = ((112, ),)
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        self.anchor_generator = AnchorsGenerator(anchor_sizes, aspect_ratios)

        self.winNum = 5

        self.classifier_cat = nn.Linear((self.winNum+1)*self.num_features, num_classes)

    def forward(self, imgs, gt_labels):
        
        self.map_origin.weight.data.copy_(self.classifier.weight.data.unsqueeze(-1).unsqueeze(-1))
        self.map_origin.bias.data.copy_(self.classifier.bias.data)

        # original branch
        b, _, h, w = imgs.size()
        fms_raw, conv5_b, conv4_out, conv3_out = self.backbone(imgs)
        raw_logits = self.classifier(self.avg(fms_raw).view(-1, self.num_features))

        # object branch
        with torch.no_grad():
            # b,c,h,w->b,nc,h,w
            crm_raw = self.map_origin(fms_raw)  # map_origin(feature_raw)相当于将feature_raw->num_classes实现了依次分类
        crm_select_raw = self.select_correspond_map(crm_raw, gt_labels)
        # using SCDA crop on class response map
        obj_coords = torch.tensor(AOLM(crm_select_raw)).cuda()
        obj_imgs = torch.zeros([b, 3, h, w]).cuda()
        for i in range(b):
            [x0, y0, x1, y1] = obj_coords[i].data.cpu().numpy()
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
            win_img = imgs[i:i + 1, :, x0:(x1 + 1), y0:(y1 + 1)]
            obj_imgs[i:i + 1] = F.interpolate(win_img, size=(h, w), mode='bilinear', align_corners=True)
        fms_obj, _, _, _ = self.backbone(obj_imgs)
        obj_logits = self.classifier(self.avg(fms_obj).view(-1, self.num_features))  # b*winNum, nc

        with torch.no_grad():
            crm_obj = self.map_origin(fms_obj)  # b,c,h,w->b,nc,h,w

        crm_select_obj = self.select_correspond_map(crm_obj, gt_labels)
        crm_on_img = F.interpolate(crm_select_obj, size=(h, w), mode='bilinear', align_corners=True)
        # scores_coords = self.APPM(crm_select_obj, self.winNum)
        anchors = self.anchor_generator((self.input_size_net, self.input_size_net), [crm_select_obj])
        scores_coords = get_topN_coords(anchors, crm_on_img, self.winNum, 0.3)

        ba, winNum, _ = scores_coords.size()
        windows_imgs = torch.zeros([b, self.winNum, 3, h, w]).cuda()
        for i in range(b):
            score_coord = scores_coords[i]
            for j in range(self.winNum):
                [x0, y0, x1, y1] = score_coord[j, 1:].data.cpu().numpy()
                x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
                win_img = obj_imgs[i:i + 1, :, x0:(x1 + 1), y0:(y1 + 1)]
                windows_imgs[i:i + 1, j] = F.interpolate(win_img, size=(224, 224), mode='bilinear', align_corners=True)

        windows_imgs1 = windows_imgs.reshape(b * self.winNum, 3, 224, 224)
        fms_win, _, _, _ = self.backbone(windows_imgs1.detach())
        wins_logits = self.classifier(self.avg(fms_win).view(-1, self.num_features))  # b*winNum, nc

        _, _, hf, wf = fms_win.shape
        fms_win = fms_win.view(b, -1, hf, wf)
        fms_cat = torch.cat((fms_obj, fms_win), dim=1)
        cat_logits = self.classifier_cat(self.avg(fms_cat).view(-1, (self.winNum+1)*self.num_features))

        
        return raw_logits, obj_logits, wins_logits, cat_logits, self.winNum

    def select_correspond_map(self, class_response_maps, gt_labels):
        b, c, h, w = class_response_maps.size()
        select_crm = []
        for i in range(b):
            cls = gt_labels[i]
            select_crm.append(class_response_maps[i, cls, :, :].detach().unsqueeze(0).unsqueeze(0))
        select_map = torch.cat(select_crm, dim=0)
        return select_map


def AOLM(fms):
    b, c, h, w = fms.size()
    A = torch.sum(fms, dim=1, keepdim=True)  # b,c,h,w->b,1,h,w
    a = torch.mean(A, dim=[2, 3], keepdim=True)  # b,1,h,w->b,1,1,1
    M = (A > a).float()  # b,1,h,w

    coords = []
    for i, m in enumerate(M):  # 解耦b
        mask_np = m.cpu().numpy().reshape(h, w)
        component_labels = measure.label(mask_np)  # 连通区域标记
        properties = measure.regionprops(component_labels)
        areas = []
        for prop in properties:
            areas.append(prop.area)
        max_idx = areas.index(max(areas))

        max_region_mask = (component_labels == (max_idx+1)).astype(int)
        # np_m = M1[i][0].cpu().numpy()
        intersection_mask = (max_region_mask == 1).astype(int)
        prop = measure.regionprops(intersection_mask)
        if len(prop) == 0:
            bbox = [0, 0, h, w]
            print('there is one image no intersection')
        else:
            bbox = prop[0].bbox

        # bbox转化成在原图中的坐标
        tl_x = bbox[0] * 32 - 1
        tl_y = bbox[1] * 32 - 1
        br_x = bbox[2] * 32 - 1
        br_y = bbox[3] * 32 - 1

        if tl_x < 0:
            tl_x = 0
        if tl_y < 0:
            tl_y = 0
        coord = [tl_x, tl_y, br_x, br_y]
        coords.append(coord)
    return coords
