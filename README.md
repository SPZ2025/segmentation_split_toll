# segmentation_split_toll

面向多粒度文本处理流程的**大颗粒度分段工具包**。每份输入作为独立的连续原文，自动决定切点数量与位置；不负责句子切分或小颗粒段落生成，也不要求输入采用统一章节结构。

## 固定规则

- **2000–8000 tokens，含端点：直接返回整段输入原文。** 此路径只加载tokenizer，不加载编码器、不进行语义分段。
- 超过8000 tokens：内部SSE＋单尺度平滑谷深＋自动动态规划，所有输出片段均为2000–8000 tokens。
- 不足2000 tokens：原样返回，状态为`below_minimum`，交上游决定如何使用；不丢弃、不擅自跨unit合并。
- 空字符串：返回`empty`与空片段列表。
- 章节、小节和Markdown标题均按普通内容保留，不强制产生切点。不同unit彼此独立。
- 每个片段都是输入的连续子串；全部片段依次拼接严格还原输入，包括BOM、空格、原始换行。

算法参数已经固定：左右各2段、3点居中平滑、MAD校准、内部权重β=1、边界权重α=1、每刀惩罚λ=0.8。模型使用`intfloat/multilingual-e5-small`，固定模型修订。没有生成式LLM调用。具体目标与处理流程见[方法说明](docs/METHOD.md)，可用`segmentation-split --profile`查看完整参数。

## 安装

Python 3.10及以上：

```bash
pip install git+https://github.com/SPZ2025/segmentation_split_toll.git
```

或者下载仓库后，在仓库根目录执行：

```bash
pip install .
```

默认只读取本地模型缓存。首次使用且本机没有模型时，加上`--download-model`，允许从Hugging Face下载所需tokenizer与模型；后续可离线运行。只下载模型文件，输入文本在本地处理。

## Python调用

```python
from segmentation_split_toll import LargeTextSegmenter

segmenter = LargeTextSegmenter()  # 默认离线；首次下载可传local_files_only=False

with open("unit.txt", encoding="utf-8", newline="") as stream:
    original = stream.read()

result = segmenter.segment(original, unit_id="unit-001")
print(result.status, result.cut_count)

for fragment in result.fragments:
    print(fragment.start, fragment.end, fragment.token_count)
    # fragment.text是原文子串，可交给后续任务

assert "".join(f.text for f in result.fragments) == original
```

多个unit独立处理，编码器在需要时复用：

```python
results = segmenter.segment_many([unit_a_text, unit_b_text])
```

默认自动选择设备；需要指定时可用`LargeTextSegmenter(device="cpu")`或`device="cuda"`。也可以传`model_path="/path/to/multilingual-e5-small"`，指向相同模型修订的完整本地副本。`batch_size=32`只调整编码批大小，不改变算法参数。

## 命令行

单个文件，输出到目录：

```bash
segmentation-split unit.md -o outputs
```

首次允许下载模型：

```bash
segmentation-split unit.md -o outputs --download-model
```

批量处理目录内的`.txt`与`.md`，保留相对目录层级：

```bash
segmentation-split inputs -o outputs --recursive
```

单文件不传`-o`时输出JSON到标准输出，也可通过标准输入集成：

```bash
segmentation-split unit.txt > unit.segments.json
cat unit.txt | segmentation-split - > unit.segments.json
```

等价入口为`python -m segmentation_split_toll`。输入默认UTF-8，可用`--encoding`指定编码。目录处理时，输出目录不能等于或包含输入目录；输出目录位于输入目录内部时会自动排除该目录。已有输出默认拒绝覆盖，明确需要时使用`--overwrite`。

## 输出格式

`unit.md`对应`unit.md.segments.json`。主要字段：

| 字段 | 含义 |
|---|---|
| `status` | `passthrough`、`segmented`、`below_minimum`或`empty` |
| `unit_id` | 调用方标识，或输入文件的相对路径 |
| `input_tokens` | 完整输入的实际token数 |
| `cut_count` | 自动产生的切点数；直返时为0 |
| `fragments` | 按原文顺序保存的片段 |
| `fragments[].start/end` | Python字符串字符索引，左闭右开；不是字节或UTF-16坐标 |
| `fragments[].text` | 完整原文子串 |
| `fragments[].token_count` | 该子串独立token化后的实际长度 |
| `fragments[].within_limits` | 是否处于2000–8000范围 |
| `refined_candidates` | 是否因原始段落无法合法分段而细化候选位置 |
| `algorithm` | 本次使用的固定算法配置与模型修订 |

计数不加特殊token或`query:`前缀，不假定各段token数相加等于整篇token数。原始段间空白归属于前一片段，完整保留。输出JSON本身的长度不参与片段长度约束。

## 无统一段落结构的输入

优先以空行形成原始段落候选，单个换行保留在段内。只有当这些位置无法组成满足长度要求的完整划分时，才细化原文候选：优先句末和行末；极长、无标点文本再使用原字符位置，形成不超过512 tokens的内部计算单元。这些单元只供算法寻找切点，**不是正式输出的小颗粒段落**，不改变最终2000–8000约束。

如仍无法满足完整划分，Python抛出`InfeasibleSegmentationError`，命令行打印错误并以非零状态退出，不会截断原文或返回伪合规结果。批量文件处理中，其他文件继续独立处理。

## 验证

```bash
pip install --no-deps -e .
python -m unittest discover -s tests -v
```

测试使用合成文本、计数器和向量，不下载模型，覆盖完整原文还原、上下限直返、短文本、CRLF/BOM、内部标题、不规则段落、无标点长文本、批处理，以及穷举所有可行划分验证自动DP。部署前的真实E5与原算法回归结果见[验证记录](docs/VALIDATION.md)。

仓库仅包含工具包、文档与测试，不包含书籍原文、实验数据、模型权重或其他粒度的分段器。
