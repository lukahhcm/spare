对照组1: “无工具直答”（无工具了解 直接推理出最终答案） 
对照组2: 有工具baseline（正常推理调用工具 不做任何剪枝 有显示reasoning） 
对照组3: 所有step删去对照组2输出的那种所有的reasoning（每一步推理完都清空） 
对照组4: 在3的基础上不全删光，所有step保留视觉特征（每一步推理完都只保留视觉特征）


prune v3实现方案：
1. 多轮工具循环中，模型输出 <think>…</think> 和 <tool_call>…</tool_call>，每轮的 reasoning 和 action block 被记录为一个 segment。
2. 模型调用 summarize_the_task 后，得到 task-state summary。visual_state 模式要求 summary 显式组织：
      - coordinates / bbox / crop id / OCR text / A* 输出
      - candidate differences
      - observations、uncertainties、next decision

3. 对所有尚未剪枝的历史 reasoning，分别计算同一模型在两种上下文下的 token 分布：
      - student view：只有 system prompt；
      - teacher view：system prompt + Summary of prior task state。

     然后对每个 token 算 top-K 近似 reverse KL。某 segment 中若归一化 KL 大于阈值 0.2 的 token 少于 2 个，则认为 summary 已经覆盖该段信息，将其剪掉。

4. V2 剪掉一个 segment 后只留原始 <tool_call> / <response> action block。

     V3 在同一个剪枝决定下，额外从被删 reasoning 中抽取证据，替换为：

     Structured visual evidence kept:
     - OCR text ... at coordinates ...
     - candidate bbox [...]
     - left/right, color, edge/texture 等局部视觉事实

     <tool_call>...</tool_call>

prune v5实现方案：
在v3的基础上改成和几个对照组一样，不把summary写进trace，仅使用这个作为额外信息辅助下一步推理；在v3的kl判断后，kl大的留下视觉推理，kl小的全删。

作为对照组，prune v5需要有“无工具直答”（无工具了解 直接推理出最终答案） 有工具baseline（正常推理调用工具 不做任何剪枝 有显示reasoning） 所有step删去所有reasoning（每一步推理完都清空） 所有step保留视觉特征（每一步推理完都只保留视觉特征）



统一按 “reasoning-only” 计算：

  - 每个工具步骤，把 assistant 输出中 <tool_call> 之前的文本视为 reasoning。
  - <tool_call> JSON、tool response、图像 token 都不计入剪枝 token。
  - 对每个 reasoning segment 记录：
      - original_token_count：初始 reasoning token 数；
      - final_retained_token_count：最终仍留在 trace 的 reasoning/evidence token 数。

  - 未剪的 segment：两者相等，贡献 0。
  - 全删 reasoning：final_retained_token_count=0。
  - 保留视觉特征 / V5：只计最终保留的确定视觉证据 token；不计 action。
  - 多次 summary：同一 segment 只按初始长度与最终表示计一次，不按 event 重复累计。
  - 不持久保留 summary，因此不额外扣 summary token；V5 的一次性 hint 不计入 retained trace。

  公式：

  \begin{equation}
  \Delta C^{m}_{j,b}
  =
  \sum_{h\in\mathcal H_{j,b}}
  \left[
  \operatorname{Tok}(R_{\mathrm{orig}}(h))
  -
  \operatorname{Tok}(\widetilde R^{m}_{\mathrm{final}}(h))
  \right].
  \end{equation}

  \begin{equation}
  r_{m,b}
  =
  100\times
  \frac{
  \sum_{j\in\mathcal D_b}\Delta C^{m}_{j,b}
  }{
  \sum_{j\in\mathcal D_b}
  \sum_{h\in\mathcal H_{j,b}}
  \operatorname{Tok}(R_{\mathrm{orig}}(h))
  }.
  \end{equation}

  其中 (r_{m,b}) 是 benchmark-level pruning ratio。工具 baseline 应为 0%，全删 reasoning 应为 100%。
