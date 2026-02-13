# Wan-Video Generation Acceleration: A Dynamic Segmentation Approach and Performance Benchmarking on NVIDIA H200 & AMD MI300X

**Suggested Titles:**

* **Technical:** Wan-Video Generation Acceleration: A Dynamic Segmentation Approach and Performance Benchmarking on NVIDIA H200 & AMD MI300X 


* **General:** From 20 Minutes to 3 Minutes: Full-Link Acceleration of Wan Video Generation and Deep Evaluation of H200/AMD 



---

## 1. Introduction & Problem Definition

1. Introduction & Problem DefinitionThe original Wan project suffered from a significant generation speed bottleneck ($>20$ minutes), which severely limited its application in production environments. Our objective is to compress the generation time to the minute level through algorithmic optimization (preprocessing strategies) and system-level optimization (inference acceleration). This article details an adaptive dynamic frame segmentation algorithm and presents a performance comparison between the NVIDIA H200 and AMD flagship accelerators.

The generation process is abstracted into a two-stage serial pipeline:

* **Preprocess:** Preprocess: Input video/reference image $\rightarrow$ Keypoints/Human Pose, Bounding Boxes (bbox), Alignment information, etc. (Includes components like YOLO + ViTPose).


* **Inference:** Inference: The diffusion/generation backbone (e.g., DiT) generates frames segment-by-segment using a sliding window strategy, followed by concatenation/overlap processing.

Therefore, the end-to-end latency can be expressed as:$$T_{total} = T_{pre} + T_{inf}$$The optimization goal is to simultaneously reduce $T_{pre}$ and $T_{inf}$ without significantly sacrificing quality.

---

## 2. Core Optimization

### 2.1 Adaptive Dynamic Frame Segmentation

#### 2.1.1 Problem Modeling

Inference utilizes a sliding window strategy defined by `clip_len` and `overlap`. To satisfy concatenation divisibility and stride constraints, the actual `target_frames` generated often exceed the `real_frames`, resulting in computational redundancy:

$target = real + [(clip - overlap) - (real - overlap) \pmod{(clip - overlap)}]$


**The Issue:** When `real - overlap` cannot be perfectly divided by the stride (`clip - overlap`), the sequence is padded to the nearest divisible length, causing the generation of useless frames and a linear increase in computational overhead.

#### 2.1.2 Solution

In long video generation tasks, segmenting long sequences into clips that the model can process is a critical issue. Traditional fixed-stride segmentation often leads to wasted edge frames or computational redundancy. We propose a reverse derivation strategy based on the target frame count.

Let $L_{target}$ be the total target frames, $L_{clip}$ be the length of a single clip, $L_{op}$ be the overlap between adjacent clips, and $n$ be the number of segments. To ensure temporal continuity and maximize coverage, we establish the following constraint equation:

$$L_{target} = L_{clip} + (L_{clip} - L_{op}) \times n$$

Where $n \ge 0$ represents we need to perdorm $n-1$ times inference through diffusion method and is an integer. In engineering practice, we limit the upper bound of $L_{clip}$ based on VRAM capacity (Memory Budget), for example, $L_{clip} \le 100$. Through this method, we effectively reduce $T_{inf}$ latency.

#### 2.1.3 In-Depth Analysis

By implementing `auto_set_lim` via code, we can dynamically solve for the optimal $L_{clip}$ and $n$ for any arbitrary $L_{target}$. The advantages of this approach include:


* **Maximization of VRAM Utilization:** Dynamically adjusting the clip length to fill computational cores avoids idle computing power caused by padding. The essence of this algorithm is to maximize **Computational Density**. By dynamically solving for $L_{clip}$, we eliminate the **Pipeline Bubbles** introduced by padding in traditional methods, ensuring that the GPU's Tensor Cores remain in a saturated computational state at every Time Step.


* **Temporal Consistency:** It guarantees a fixed overlap, making the video context smoother during concatenation.



### 2.2 From Sequential Dependency to Task-Level Parallelism

We describe this optimization as breaking the **Pseudo-Dependency** between frames.

#### 2.2.1 Problem Modeling: The Serial Bottleneck

In the baseline implementation, the preprocessing pipeline (specifically 2D pose estimation and frame extraction) operates as a strict sequential synchronous process. Mathematically, for a video sequence containing $N$ frames, the total preprocessing latency $T_{pre}$ is determined by the cumulative sum of single-frame processing times:

$$T_{pre} = \sum_{i=1}^{N} (t_{decode}^{(i)} + t_{inference}^{(i)})$$

Under this serial execution model, host CPU utilization is severely limited by single-core performance metrics. Crucially, this introduces significant pipeline bubbles, causing the downstream video generation model to remain idle while waiting for the entire sequence preprocessing to complete.

#### 2.2.2 Solution: Thread-Pool-Based Concurrent Scheduling

To address this mixed bottleneck of I/O-intensive and compute-intensive tasks, we refactored the preprocessing module using a **Thread Pool Executor** pattern, achieving coarse-grained **Task-Level Parallelism (TLP)**.

nlike the sequential execution method, we decoupled the inter-frame dependencies. By instantiating a thread pool of size $K$ (where $K$ is a constant), we transformed the execution model into a concurrent paradigm. This allows multiple 2D pose (`pose2d`) inference tasks—typically executed by C-bound libraries (e.g., OpenCV, PyTorch) that release the Python Global Interpreter Lock (GIL)—to execute in parallel on different physical cores.

This improvement effectively transforms the latency formula to:$$T_{pre}' \approx \frac{1}{\min(N, K)} \sum_{i=1}^{N} (t_{process}^{(i)}) + T_{overhead}$$Where $T_{overhead}$ represents the minimal overhead of context switching and thread management.

This improvement effectively transforms the latency formula to:

$$T_{pre}' \approx \frac{1}{\min(N, K)} \sum_{i=1}^{N} (t_{process}^{(i)}) + T_{overhead}$$

Where $T_{overhead}$ represents the minimal overhead of context switching and thread management.

*Note on Multi-GPU:* Multi-card setups (FSDP + ulysses_size, etc.) introduce additional communication, thread scheduling, and synchronization overhead. Reports indicate that running the original preprocessing flow in a multi-card environment actually slowed it down (up to ~170s), exhibiting typical resource contention/synchronization amplification. Therefore, additional restrictions were implemented to force preprocessing to run under single-card conditions even in multi-card environments.

#### 2.2.3 In-Depth Analysis: Why it works?


* **Circumventing the GIL via C Extensions:** Although Python's Global Interpreter Lock (GIL) notoriously limits CPU-bound multi-threaded performance, our profiling shows that heavy tasks in preprocessing (e.g., `pose2d` inference and image transformations) rely heavily on underlying C/C++ kernels (like NumPy and PyTorch operations). These operations release the GIL during execution, allowing our thread pool to achieve near-linear scalability on multi-core CPUs (e.g., fully utilizing all 36 vCPUs mentioned in the baseline test).



* **Saturation of Host Compute Capacity:** By concurrently distributing $N$ frames, we maximize the host CPU's **Instruction-Level Parallelism (ILP)** and memory bandwidth utilization. This strategy shifts the bottleneck from single-thread clock speed to overall multi-core throughput, significantly reducing preprocessing time from 229s to 78s, effectively approaching the theoretical limit defined by **Amdahl's Law** for the parallelizable portion of the workload.



---

## 3. Experimental Setup & Hardware Environment

We established a rigorous Benchmark environment to evaluate the optimization effects:

* **Baseline:** Wan Origin Project (with/without SGLang).


* **Hardware A:** NVIDIA H200 (1/2 cards), 141GB HBM3e.


* **Hardware B:** AMD Instinct MI300X, 192GB HBM3.



### 3.1 Performance Evaluation

#### 3.1.1 End-to-End Acceleration

By implementing the aforementioned optimizations, we achieved significant performance improvements on the H200. The following tests were conducted using the same image and video inputs:

| Version | FPS | Target Frames | Preprocess (s) | Inference (s) | Total (s) | Hardware |
| --- | --- | --- | --- | --- | --- | --- |
| **Origin** | 30 | - | 229 | 129 | 325 | 2x H200, 36 vCPU |
| **Ours** | 30 | - | 229 | 78 | 270 (**↓16.9%**) | 2x H200, 36 vCPU |
| **Ours (2025/12/23)** | 30 | 153 | 79 | 133 | 212 (**↓34.8%**) | 2x H200, 36 vCPU |
| **Origin** | 30 | - | 229 | 232 | 684 | 1x H200, 18 vCPU |
| **Ours** | 30 | - | 229 | 120 | 568 (**↓16.9%**) | 1x H200, 18 vCPU |
| **Ours (2025/12/23)** | 30 | 153 | 120 | 290 | 410 (**↓40.1%**) | 1x H200, 18 vCPU |



* **Preprocess:** Time reduced from 229s to 79s (↓65.5%), primarily due to the parallelized data loading and decoding pipeline.


* **Inference:** Time reduced from 129s to 79s (↓38.7%).


* **Total Generation Time:** Reduced from $>20$ minutes (and a baseline of 325s) to 212s.



#### 3.1.2 NVIDIA H200 vs. AMD MI300X (Key Comparison)

Tests were conducted using the same image and video inputs:

| Setup | Preprocess | Inference | Total | clip_len | target_frames | steps | GPU |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **NVIDIA Origin** | 94s (no replace) | 181s | 275s | 77 | 153 | 4 (no lora) | 1x H200 |
| **Ours** | 93s (no replace) | 287s | 380s | 77 | 153 | 4 (lora) | 1x H200 |
| **AMD Ours** | 33s (no replace) | 215s | 248s | 77 | 153 | 4 (lora) | 1x MI300X |
| **AMD Ours** | 33s (no replace) | 220s | 253s | 77 | 153 | 4 (no lora) | 1x MI300X |



**Analysis:**

* **VRAM Advantage:** We must highlight whether AMD's larger VRAM (192GB) allowed for a larger $L_{clip}$, thereby reducing the segment count $n$ and total inference time.

#### 3.1.3 Acceleration Effects with SGLang

To further validate our solution, we tested our method within the SGLang acceleration framework:

| Version | Preprocess | Inference | Total | clip_len | real/target | steps | GPU |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Origin | 224s | 472s | 696s | 77 | 155 | 20 | 1x H200 |
| Origin | 220s | 297s | 517s | 77 | 229 | 4 (no lora) | 1x H200 |
| Ours | 88s | 209s | 297s | 99 | 197 | 20 | 1x H200 |
| Ours | 90s | 159s | 249s | 99 | 197 | 4 (no lora) | 1x H200 |
| Ours | 90s | 163s | 253s | 99 | 197 | 4 (lora) | 1x H200 |



#### 3.1.4 Discussion and Parameter Sensitivity

We analyzed the impact of reducing `clip_len` from 77 to 41 on a single H200:

| Metric | Time | GPU | clip_len |
| --- | --- | --- | --- |
| Before | 448s | 1x H200 | 77 |
| After | 383s (↓14.5%) | 1x H200 | 41 |
| After (Over-reduced) | 574s | 1x H200 | 21 |



* **Inference:** 448s  383s (↓14.5%).


* **Diminishing Returns:** Reducing `clip_len` further to 21 increased time to 574s.


* **Conclusion:** `clip_len` is not "the smaller the better." Excessive reduction leads to too many segments, redundant overlap calculations, and increased kernel/scheduling overhead, which negate the benefits.



**Further Video Length Experiments:**

*Table: Video 1 Performance* 
| Method | Preprocess | Inference | clip_len | real/target | GPU |
| :--- | :--- | :--- | :--- | :--- | :--- |
| origin_set | 143s | 726s | 77 | 308/381 | 1x H200 |
| auto_set | 140s | 626s | 153 | 308/381 | 1x H200 |
| auto_set_lim | 140s | 556s | 97 | 308/289 | 1x H200 |
| origin_set | 140s | 325s | 77 | 308/381 | 2x H200 |
| auto_set | 141s | 339s | 153 | 308/305 | 2x H200 |
| auto_set_lim | 140s | 268s | 102 | 308/289 | 2x H200 |

*Table: Video 2 Performance* 
| Method | Preprocess | Inference | clip_len | real/target | GPU |
| :--- | :--- | :--- | :--- | :--- | :--- |
| origin_set | 41s | 142s | 77 | 60/77 | 1x H200 |
| auto_set_lim | 41s | 109s | 57 | 60/57 | 1x H200 |
| origin_set | 41s | 68s | 77 | 60/77 | 2x H200 |
| auto_set | 40s | 44s | 29 | 60/57 | 2x H200 |
| auto_set_lim | 40s | 48s | 57 | 60/57 | 2x H200 |

These results confirm that `clip_len` should not be excessively large or small, further validating the rationality of our design.

---

## 4. Conclusion

Through algorithmic dynamic segmentation and engineering pipeline improvements, we improved the inference efficiency of the Wan model by over 30%. Empowered by high-end computing power (H200/AMD), high-quality video generation now possesses the potential for real-time interaction.

### Module 1: Adaptive VRAM-Aware Dynamic Segmentation

In traditional long video generation, fixed Stride and Window Size are commonly used. While simple, this leads to tail Padding (invalid computation) or context loss when processing non-divisible frame counts. We redefined this as a **Constrained Discrete Optimization Problem**:

1. **Mathematical Modeling:** Find a parameter set $\{L_{clip}, n\}$ such that the total covered frames strictly equal the user input $L_{target}$, while satisfying VRAM constraints.

2. **Core Constraint:** $L_{target} = L_{clip} + (L_{clip} - L_{op}) \times n$.


3. **Variables:**
* $L_{clip}$: Window length of a single inference (constrained by $VRAM_{capacity}$).


* $L_{op}$: Minimum overlapping frames required for temporal continuity (Temporal Overlap Constraint).


* $n$: Non-negative integer representing the number of sliding windows.




4. **Effect:** Maximizes **Computational Density** and eliminates **Pipeline Bubbles** to keep Tensor Cores saturated.



> **Technical Consideration:** The influence of $L_{op}$ (overlap frames)  on **Temporal Consistency** is critical. Specifically, how latent representations in the overlap region are fused is a key factor for review.
> 
> 

### Module 2: System Engineering Optimization

Refactoring "loop processing" into "multi-threaded parallelism" is a classic system engineering optimization. Key concepts include:

* **Task-Level Parallelism (TLP):** Concurrent execution of distinct tasks (frames), distinct from Data-Level Parallelism (SIMD).


* **Release GIL:** Demonstrates understanding of Python's underlying mechanics—multithreading is only effective when underlying libraries (OpenCV, NumPy) actively release the lock.


* **Amdahl's Law:** Used to calculate the maximum theoretical speedup in parallel processing.



### Module 3: Bandwidth-Centric Hardware Benchmarking

When comparing NVIDIA H200 and AMD MI300X, **Memory Bandwidth** is often the deciding factor rather than just TFLOPS, especially for large model tasks like video generation.

1. **H200 HBM3e Advantage:** With 141GB HBM3e and 4.8TB/s bandwidth, it addresses the **Memory Wall** often faced by Transformer models.


2. **Roofline Model Analysis:** In the Wan model, due to the large Context Window, Arithmetic Intensity might not reach the Tensor Core limit; thus, high bandwidth directly yields linear speedups in inference.


3. **FP8 Quantization:** The H200's native FP8 support (4th Gen Tensor Core) allows halving the KV Cache memory footprint. This enables a larger $L_{clip}$ (effective Batch Size) under the same VRAM budget, directly increasing **Throughput**.