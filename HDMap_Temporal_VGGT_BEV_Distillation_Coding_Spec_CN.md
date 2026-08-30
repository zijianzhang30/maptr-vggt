# 实现说明：Temporal VGGT-BEV Distillation for MapTR / MapTRv2

## 0. 目标

为 MapTR / MapTRv2 实现一个 **训练阶段使用的多帧 VGGT-BEV 蒸馏模块**。

核心流程：

```text
连续多帧多视角图像
        ↓
Frozen VGGT
        ↓
VGGT feature + point map / depth
        ↓
Point-to-BEV Pooling
        ↓
Ego-motion 对齐的时序 BEV 融合
        ↓
Teacher BEV feature
        ↓
蒸馏 MapTR / MapTRv2 encoder 的 BEV feature
```

推理阶段必须保持原始 MapTR / MapTRv2 不变：

```text
当前帧多视角图像
        ↓
原始 MapTR / MapTRv2 encoder
        ↓
原始 MapTR / MapTRv2 decoder/head
        ↓
vectorized map
```

推理阶段不使用：

```text
VGGT
teacher branch
Point-to-BEV Pooling
Temporal Fusion
Auxiliary head
Projection heads
```

---

## 1. 保留和新增内容

### 1.1 保留

```text
原始 MapTR / MapTRv2 decoder/head
原始 vector map loss
原始推理流程
```

### 1.2 训练阶段新增

```text
Frozen VGGT teacher
Point-to-BEV Pooling 模块
Temporal BEV Fusion 模块
BEV feature distillation loss
可选 auxiliary raster head
```

### 1.3 推理阶段删除

```text
VGGT
VGGT feature cache
Point-to-BEV Pooling
Temporal BEV Fusion
用于蒸馏的 projection heads
Auxiliary raster head
```

---

## 2. 输入

对于当前训练样本 `t`：

```text
Student input:
    当前帧多视角图像 I_t^{1:N}

Teacher input:
    连续多帧多视角图像 {I_{t-L:t+L}^{1:N}}
```

第一版推荐：

```text
teacher frames = {t-2, t-1, t, t+1, t+2}
num_teacher_frames = 5
```

必须提供的 metadata：

```text
camera intrinsics
camera extrinsics
每一帧的 ego pose
BEV range and resolution
GT vector map
```

可选 metadata：

```text
VGGT confidence
road mask
depth confidence
```

---

## 3. VGGT 特征缓存

VGGT 冻结，不参与训练。

由于 VGGT 前向较重，建议先离线预计算并缓存 VGGT 输出。

对每一帧、每一个相机缓存：

```text
vggt_feat:       [C_v, H_v, W_v]
point_map/depth: [3, H_v, W_v] 或 [H_v, W_v]
confidence:      [H_v, W_v]，可选
```

建议缓存 key：

```text
scene_id / sample_token / camera_name / frame_timestamp
```

训练 dataloader 需要根据当前样本加载 teacher frames 对应的 VGGT cache。

---

## 4. Point-to-BEV Pooling

## 4.1 输入

对于 teacher frame `τ` 和 camera `k`：

```text
Z_{τ,k}: VGGT feature, [C_v, H_v, W_v]
P_{τ,k}: point map or depth
K_k: camera intrinsics
T_{τ,k}: camera-to-ego transform
C_{τ,k}: confidence, optional
```

---

## 4.2 将 image token lift 到 3D

如果使用 depth：

```text
p_cam = depth[u,v] * inv(K) @ [u, v, 1]
p_ego = T_cam_to_ego @ p_cam
```

如果使用 VGGT point map：

```text
p_ego = point_map[u,v]
```

注意：

```text
p_ego 必须和 MapTR 的 BEV 坐标系保持一致
```

---

## 4.3 将 3D point 转成 BEV cell

给定 BEV 范围：

```text
x_min, x_max
y_min, y_max
bev_resolution
```

计算：

```text
b_x = floor((x - x_min) / bev_resolution)
b_y = floor((y - y_min) / bev_resolution)
```

只保留合法 BEV cell：

```text
0 <= b_x < W_bev
0 <= b_y < H_bev
```

---

## 4.4 将 VGGT feature pool 到 BEV

对于每个有效 BEV cell，将落入该 cell 的 VGGT features 做 pooling。

第一版实现：

```text
mean pooling
```

推荐实现：

```text
confidence-weighted mean pooling
```

公式：

```text
F_τ^{bev}(b_x,b_y) = sum_i(w_i * z_i) / (sum_i(w_i) + eps)
```

默认权重：

```text
w_i = 1
```

可选权重：

```text
w_i = VGGT confidence * valid_depth_mask
```

输出：

```text
F_τ^{bev}: [C_t, H_bev, W_bev]
V_τ^{bev}: valid mask, [1, H_bev, W_bev]
```

如果 `C_v != C_t`，加一个小的 projection 层：

```text
1x1 Conv / Linear projection
```

---

## 5. Ego-motion Alignment

对于每个 teacher frame `τ`，需要将它的 BEV teacher feature 对齐到当前帧 `t`。

输入：

```text
F_τ^{bev}
V_τ^{bev}
ego pose at τ
ego pose at t
```

计算变换：

```text
T_{τ→t}
```

warp：

```text
F_{τ→t}^{bev} = warp_bev(F_τ^{bev}, T_{τ→t})
V_{τ→t}^{bev} = warp_bev(V_τ^{bev}, T_{τ→t})
```

实现建议：

```text
grid_sample + bilinear interpolation
```

对齐完成后，所有 teacher BEV features 都应该位于当前帧 `t` 的 ego 坐标系下。

---

## 6. Temporal BEV Fusion

输入：

```text
{F_{τ→t}^{bev}} for τ in teacher frames
{V_{τ→t}^{bev}} for τ in teacher frames
```

第一版实现：

```text
valid-mask weighted average
```

公式：

```text
F_T^{bev} = sum_τ(V_{τ→t}^{bev} * F_{τ→t}^{bev}) / (sum_τ(V_{τ→t}^{bev}) + eps)
```

输出：

```text
F_T^{bev}: teacher BEV feature, [C_t, H_bev, W_bev]
M_T^{valid}: teacher valid mask, [1, H_bev, W_bev]
```

后续可选升级：

```text
temporal attention over aligned BEV features
```

第一版先不要实现 temporal attention，先保证 weighted average 版本跑通。

---

## 7. Student BEV Feature

从 MapTR / MapTRv2 encoder 中取 decoder/head 之前的 BEV feature：

```text
F_S^{bev}: [C_s, H_bev, W_bev]
```

如果 student BEV 分辨率和 teacher BEV 分辨率不一致，需要 resize。

推荐：

```text
resize teacher F_T^{bev} 到 student BEV resolution
```

---

## 8. Projection Heads

teacher 和 student 的 channel 可能不同。

使用 projection heads 对齐通道：

```text
Proj_T: C_t → C_d
Proj_S: C_s → C_d
```

建议结构：

```text
1x1 Conv
Norm
ReLU
1x1 Conv
```

得到：

```text
F_T_proj = Proj_T(F_T^{bev})
F_S_proj = Proj_S(F_S^{bev})
```

蒸馏时 teacher target 需要 detach：

```text
F_T_proj = stop_gradient(F_T_proj)
```

---

## 9. Map-aware Distillation Mask

不要对全 BEV 区域蒸馏。

从 GT vector map 构造 mask：

```text
Y_map = rasterize GT map into BEV
M_map = dilate(Y_map)
```

推荐类别：

```text
divider
boundary
pedestrian_crossing
```

mask shape：

```text
M_map: [1, H_bev, W_bev]
```

最终蒸馏 mask：

```text
M = M_map * M_T^{valid}
```

如果不使用 teacher valid mask：

```text
M = M_map
```

---

## 10. Distillation Loss

使用 normalized feature distillation。

---

## 10.1 L1 feature loss

```text
F_S_norm = normalize(F_S_proj, dim=channel)
F_T_norm = normalize(F_T_proj, dim=channel)

L_l1 = mean_masked(abs(F_S_norm - F_T_norm), M)
```

---

## 10.2 Cosine loss

```text
L_cos = mean_masked(1 - cosine_similarity(F_S_proj, F_T_proj), M)
```

---

## 10.3 最终 feature distillation loss

```text
L_feat = L_l1 + beta * L_cos
```

第一版推荐：

```text
beta = 1.0
```

---

## 11. Auxiliary Raster Head

从 student BEV feature 后面接一个训练时 auxiliary head：

```text
P_S = AuxHead(F_S^{bev})
```

输出：

```text
P_S: [C_map, H_bev, W_bev]
```

类别：

```text
divider
boundary
pedestrian_crossing
```

target：

```text
Y_map = rasterize GT vector map
```

loss：

```text
L_heat = BCE(P_S, Y_map) + Dice(P_S, Y_map)
```

推理阶段删除这个 head。

---

## 12. Total Loss

使用原始 MapTR / MapTRv2 loss，加蒸馏 loss：

```text
L_total = L_maptr + λ_feat * L_feat + λ_heat * L_heat
```

初始权重建议：

```text
λ_feat = 0.1
λ_heat = 1.0
```

建议对 `λ_feat` 做 warmup：

```text
λ_feat 在前 10% training steps 从 0 线性增加到目标值
```

如果训练不稳定，降低：

```text
λ_feat = 0.05
```

---

## 13. 训练流程

### Step 1：准备 baseline

训练或加载稳定的 MapTR / MapTRv2 baseline。

---

### Step 2：缓存 VGGT 输出

离线运行 frozen VGGT，缓存 teacher clips 需要的所有帧。

缓存：

```text
VGGT features
point map / depth
confidence, optional
```

---

### Step 3：训练时构造 teacher BEV

每个 batch：

```text
load cached VGGT outputs of teacher frames
point-to-BEV pool each frame
ego-motion align all teacher frames to current frame
temporal fuse to get F_T^{bev}
```

---

### Step 4：训练 student

当前帧输入 MapTR / MapTRv2：

```text
I_t → MapTR encoder → F_S^{bev} → decoder/head → vector map prediction
```

计算：

```text
L_maptr
L_feat
L_heat
```

反向传播只更新：

```text
MapTR / MapTRv2 student
Proj_S
Proj_T if trainable
AuxHead
```

不要更新 VGGT。

---

## 14. 推理流程

推理必须保持原始 MapTR / MapTRv2：

```text
I_t
    ↓
MapTR / MapTRv2 encoder
    ↓
MapTR / MapTRv2 decoder/head
    ↓
vectorized map
```

不要调用：

```text
VGGT
VGGT cache
Point-to-BEV Pooling
Temporal Fusion
AuxHead
Projection heads
```

---

## 15. 必要消融开关

实现时预留以下配置：

```text
use_vggt_teacher: true / false
use_temporal_teacher: true / false
use_map_mask: true / false
use_aux_heatmap: true / false
teacher_num_frames: 3 / 5 / 7
teacher_frame_mode: history / bidirectional
pooling_type: mean / confidence_weighted
distill_loss: l1 / cosine / l1+cosine
lambda_feat
lambda_heat
```

主要对比：

```text
1. MapTRv2 baseline
2. + multi-frame VGGT-BEV feature distillation
3. + map-aware mask
4. + auxiliary raster head
5. full model
```

---

## 16. 建议文件 / 模块结构

建议新增目录：

```text
projects/
  mmdet3d_plugin/
    maptr/
      distill/
        vggt_cache_dataset.py
        point_to_bev_pooling.py
        bev_warp.py
        temporal_bev_fusion.py
        distill_losses.py
        aux_raster_head.py
```

主要模块：

```text
PointToBEVPooling
BEVFeatureWarper
TemporalBEVFusion
BEVDistillationLoss
AuxRasterHead
```

---

## 17. 实现注意事项

1. 确保 BEV 坐标系和 MapTR 完全一致。
2. 确保 teacher BEV resolution 和 student BEV resolution 一致。
3. 蒸馏时 teacher feature 必须 detach。
4. 只在 map-aware mask 内做蒸馏。
5. 第一版先使用 mean pooling 和 weighted average temporal fusion。
6. 简单版本跑通后，再加入 confidence weighting。
7. 对 `λ_feat` 使用 warmup。
8. 推理代码路径必须保持不变。
9. 训练前先可视化 teacher BEV feature。
10. 可视化 `M_map`、`F_T^{bev}` activation 和 auxiliary heatmap。

---

## 18. 最小可运行版本

第一版最小实现包含：

```text
1. 加载 cached VGGT feature + point map / depth
2. Point-to-BEV mean pooling
3. Ego-motion BEV warp
4. 多帧 weighted average fusion
5. Map-aware masked feature distillation
6. Auxiliary raster head
7. 原始 MapTR loss 保持不变
```

第一版先跳过：

```text
temporal attention
teacher trainable head
teacher confidence prediction
复杂 feature alignment
uncertainty modeling
```

---

## 19. 最终训练图

```text
Teacher:
cached frozen VGGT outputs from multi-frame multi-view images
        ↓
point-to-BEV pooling
        ↓
ego-motion alignment
        ↓
temporal BEV fusion
        ↓
F_T^{bev}

Student:
current multi-view images
        ↓
MapTR / MapTRv2 encoder
        ↓
F_S^{bev}
        ↓
MapTR / MapTRv2 decoder/head

Loss:
L_total = L_maptr + λ_feat L_feat + λ_heat L_heat

Inference:
original MapTR / MapTRv2 only
```
