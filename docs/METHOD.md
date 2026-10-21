# 大颗粒度连续文本分段

## 输入与直接返回

一份输入对应一段独立连续文本。先对完整输入精确计数；2000–8000 tokens时直接返回输入本身。不根据章节名或小节标题预切分，不将多个unit拼接。

超过上限时才进入语义分段。空行提供候选位置；原始位置无法形成合法划分时，再细化原文中的候选位置。所有计数与输出均针对真实原文子串，输出原文不经tokenizer解码重建。

## 评分与全局目标

采用E5归一化段落向量。左右各取2个计算单元的平均向量，计算余弦相似度；进行3点居中移动平均，端部只平均有效位置。分别沿曲线不下降方向寻找左右邻近峰顶，遇下降停止。谷深为左右峰顶相对当前位置的落差之和。

在当前输入内，对有效谷深作MAD校准：

\[
E_i=\tfrac13\operatorname{clip}\left(\frac{v_i-\operatorname{median}(v)}{1.4826\operatorname{MAD}(v)+10^{-6}},-3,3\right).
\]

完整窗口不足的位置证据为0，仍允许切分。内部代价为段落向量相对片段均值的SSE，按计算单元等权。自动动态规划最小化：

\[
\sum C(a,b)-\sum_{g\in B}E_g+0.8|B|,\qquad2000\le T(a,b)\le8000.
\]

切点数量是最优划分的结果，无须输入K。每个终点比较全部可达起点，不假定token计数可加或单调，不使用PELT剪枝。同分优先切点较少的划分，再保留最先枚举的末片段起点。

## 编码口径

模型固定为`intfloat/multilingual-e5-small`，修订`614241f622f53c4eeff9890bdc4f31cfecc418b3`。编码时采用对称`query: `前缀；超模型长度的计算单元分块编码，按块token数加权平均后L2归一化。编码分块只影响向量计算。

长度计数使用同模型tokenizer，不加特殊token，不加编码前缀。计数、模型权重均可从本地缓存读取；只有显式允许下载时才访问模型托管服务。

## 方法来源

- Hearst (1997), [Text Tiling](https://aclanthology.org/J97-1003/)：相似度曲线与谷深思想。本工具采用稠密语义向量变体。
- [NLTK TextTiling实现](https://www.nltk.org/_modules/nltk/tokenize/texttiling.html)：邻近峰顶与谷深计算参考。
- [ruptures CostL2](https://centre-borelli.github.io/ruptures-docs/user-guide/costs/costl2/)：内部SSE分段代价。本工具的自动递推独立实现，不依赖ruptures。
- Wang et al. (2024), [Multilingual E5 Text Embeddings](https://arxiv.org/abs/2402.05672)：编码模型来源。
