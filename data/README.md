# Released datasets

本目录提供项目生成并用于已报告实验的数据集，不包含原始CSDS/DCH-2文件。

## 数据目录

| 目录 | 内容 | 规模 | Manifest SHA256 |
|---|---|---:|---|
| `ecommerce/domain_train_v1_3_2_zh/` | 10类售后任务的SFT轨迹与Decision/Parameter/Response候选 | 12K / 5K | `5d9fd79139f60e1c57e88ca519b0f58ed8cfe4db9993cfac1eea1d8fa6b0215d` |
| `ecommerce/dpo_v1_4_rollout_quality_screen_800_v2/` | 首次行为分歧构造的主实验偏好集 | 720 / 80 | `37915ba0790e8c291d3f905510facb2bea4bb2c8ffbf7fda8a14bda321e80bb2` |
| `ecommerce/dpo_v1_4_rollout_quality_composition_matched_v1/` | 多粒度与Response-only等量组成消融 | 173 / 19 | `557255713e255ab72385e646c00327645b35b8e14b148a7e99ed287f8d9db366` |
| `ecommerce/formal_test_v2/` | 300 IID、150 Compositional、150 Challenge | 600 | `76285f06b871fb40f8bcad69986c9ea99c203df60fe23be567e3d7318d0ba74a` |

每个目录内的`manifest.json`记录生成版本、split规模、数据hash、场景/粒度组成和审计结果。训练脚本默认读取这些相对路径，无需重新生成数据。

## 从源数据重建

如需验证完整数据管线，可分别从以下项目页面获取源数据：

- CSDS：https://github.com/xiaolinAndy/CSDS
- DCH-2：https://dialeval-2.github.io/DCH-2/

使用`prepare_public_pilot.py`完成格式适配后，运行`build_domain_pilot_v1.py`即可重建SFT轨迹、偏好候选、审计报告和manifest。所有派生样本保留`parent_id`，用于父样本级切分、贡献上限和跨split泄漏审计。
