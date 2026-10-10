# CUDA 可用性评估

检查时间：2026-10-07。本次检查了本机 GPU、专用 `Gensim` 环境、项目调用路径、固定版本源码和官方文档。结论是：当前七种方法没有能直接切换到 CUDA 的主要训练环节，继续使用 CPU。向量汇总和归一化等外围计算可以另行改写，但尚无本工程的 CUDA 性能测量支持这种改动，因此按“有合适位置尽量使用，没有则不强行改”的要求保留当前实现。

## 本机与环境证据

[实际检查记录](../artifacts/cuda_audit.json)来自[检查脚本](../tools/check_cuda.py)，可以在项目根目录重跑：

```powershell
conda run -n Gensim python tools/check_cuda.py
```

| 项目 | 实测值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5070 |
| 驱动 | 616.64 |
| 显存 | nvidia-smi 报告 12,227 MiB |
| 计算能力 | 12.0 |
| 系统与 Python | Windows 11，Python 3.12.15 |
| 模型依赖 | Gensim 4.4.0、fasttext-wheel 0.9.2、scikit-learn 1.7.2 |
| 数值依赖 | NumPy 1.26.4、SciPy 1.16.3 |
| 其他后端 | 此环境没有安装 PyTorch、CuPy、cuML |
| LinearSVC 的 Array API 支持标记 | false |

GPU 被驱动识别不等于当前模型已能使用 CUDA。此次没有安装 CUDA Toolkit 或 GPU Python 库，没有执行 CUDA 内核基准；是否安装其他 GPU 库也不是判断现有模型支持情况的依据。决定依据是下面的实际计算路径。

## 按计算环节检查

项目在[models.py](../gensim_lab/models.py)中创建表示模型、训练 LinearSVC 并完成预测。构造函数签名和环境检测结果保存在上述 JSON 中；接口检查结合源码检查使用，不能仅以缺少 `device` 参数判断支持情况。

| 环节 | 当前实现与 CUDA 支持情况 | 决定及依据 |
| --- | --- | --- |
| 多数类基线 | DummyClassifier，统计训练标签并输出常量 | 计算量极小，保留 CPU |
| TF-IDF | Gensim Dictionary、TfidfModel，SciPy 稀疏矩阵 | 当前实现是 CPU；更换为 GPU 稀疏文本流程需要另行实现 |
| LSI | Gensim LsiModel，NumPy、SciPy 稀疏乘法与 SVD | 固定实现没有 CUDA 后端；`distributed` 是分布式处理选项，并非 GPU 选项 |
| Word2Vec | Gensim Cython 训练内核及主机 BLAS | 固定版本没有 CUDA 训练路径，`workers` 控制 CPU 线程 |
| Doc2Vec | Gensim doc2vec_inner 的 DBOW/DM 训练与推断 | 固定版本没有 CUDA 训练或推断路径 |
| Gensim FastText | fasttext_inner Cython 内核及主机向量运算 | 固定版本没有 CUDA 训练路径，保留现有子词及 OOV 实现 |
| 原生 FastText | C++ 模型，经 Python 绑定调用，std::thread 训练 | 官方明确仅使用 CPU；`thread` 不是 GPU 设置 |
| 五种表示后的分类器 | sklearn LinearSVC 经 _fit_liblinear 调用 liblinear.train_wrap | 1.7.2 的该估计器不支持 Array API GPU 分派，输入换成 CUDA 数组不能直接让训练上 GPU |
| 文章向量均值、TF-IDF 加权和 L2 归一化 | 主机词表查找、NumPy 累加和 sklearn normalize | 数值归约可改写为 CuPy/PyTorch 批处理，但词向量训练仍在 CPU，还需传输和整理数据；收益未经测量，暂不改写 |
| 清洗、去重、划分和报表 | 文本处理、SimHash、组划分、文件 I/O 和图表 | 当前流程没有直接的 CUDA 后端，不为这些环节新增 GPU 依赖 |

## 固定版本源码与官方依据

- Gensim 4.4.0：[word2vec_inner.pyx](../references/sources/gensim-4.4.0/models/word2vec_inner.pyx)第 23–34 行从 SciPy BLAS 导入主机函数指针；第 497、596 行为 Skip-gram/CBOW 训练入口。[fasttext.py](../references/sources/gensim-4.4.0/models/fasttext.py)第 539 行调用 [fasttext_inner.pyx](../references/sources/gensim-4.4.0/models/fasttext_inner.pyx)的 `train_batch_any`。[doc2vec.py](../references/sources/gensim-4.4.0/models/doc2vec.py)第 80 行导入 doc2vec_inner，第 399 行开始执行训练任务。
- Gensim LSI：检查本环境的 `Lib/site-packages/gensim/models/lsimodel.py`，第 66–72 行导入 NumPy/SciPy，第 308、1031 行调用 `scipy.linalg.svd`。这些路径不会因 NVIDIA 驱动存在而自动调用 CUDA。库的设计与实现背景见[Gensim 官方介绍](https://radimrehurek.com/gensim/intro.html)。
- FastText 0.9.2：[归档 fasttext.cc](../references/sources/fastText-0.9.2/src/fasttext.cc)第 784–787 行创建 `std::thread` 并运行 `trainThread`。官方 FAQ 的“Can we run fastText program on a GPU?”说明 CPU 支持情况：[FastText FAQ](https://fasttext.cc/docs/en/faqs.html)。
- scikit-learn 1.7.2：[LinearSVC 文档](https://scikit-learn.org/1.7/modules/generated/sklearn.svm.LinearSVC.html)明确其 liblinear 实现。[Array API 支持列表](https://scikit-learn.org/1.7/modules/array_api.html)没有 LinearSVC；本环境 `get_tags(LinearSVC()).array_api_support` 实测为 false。[固定版本 _base.py](https://github.com/scikit-learn/scikit-learn/blob/1.7.2/sklearn/svm/_base.py#L1230)第 1230 行调用 liblinear 训练。

## 为什么没有替换后端

cuML 等库可以提供部分算法的 GPU 实现，但它们是另一套实现，并不能通过给当前 Gensim 或 sklearn LinearSVC 加一个 CUDA 参数来启用。RAPIDS 的 Windows 支持使用 WSL2，需要独立的 Linux 环境，见[官方安装说明](https://docs.rapids.ai/install/)。这样的替换需要重新核对求解器、类别权重、参数含义、随机性、模型序列化和性能；不能沿用现有结果并声称来自 CUDA 实验。

现有结果中，一月 12 类 TF-IDF 的整个拟合和选参约 1.5 秒，LSI 约 2.7 秒；规模扩展后的 Gensim FastText 约 406.5 秒，见[最终实验报告](最终实验报告.md)和[时间扩展实验](时间扩展实验.md)。这些是 CPU 整体拟合耗时，并未分解训练与外围计算耗时，也没有测得 CUDA 加速比。最耗时的 FastText 训练恰好缺少原生 CUDA 路径；仅迁移外围运算的整体收益不能据此量化。

本次只新增可重跑的硬件/API 检查及本说明，并在 README 增加入口。模型、实验配置、训练代码、锁定依赖和原有实验结果均保持原样；已有保存加载与重训一致性证据仍对应原来的 CPU 实验。没有为 CUDA 检查重跑完整实验，也没有将 CPU 测量标记为 GPU 测量。
