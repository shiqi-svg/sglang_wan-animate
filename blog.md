# # From 20 Minutes to 3 Minutes: Full-Pipeline Acceleration of Wan Video Generation and Deep Evaluation of H200/AMD MI300X



---

## 1. Introduction

Wan is a diffusion‑based generative model, designed to create high‑quality images through iterative denoising. It showcases strong performance in visual generation tasks, particularly in producing detailed and stylistically consistent outputs.

The Wan model faces a major bottleneck in generation speed—often exceeding 20 minutes—which severely constrained its use in production environment. The goal of our study is to reduce its latency to the minute level through both algorithmic optimization (e.g., adaptive frame-segmentation) and system‑level enhancement (e.g., parallelism management). This article presents our efforts that improves each stage of the video generation pipeline in Wan. Furthermore, we provide performance study across NVIDIA H200 and AMD MI300x (two flagship accelerators), giving performance analysis from the architecture persepctive. 

The generation process in Wan can be summarized as a two-stage pipeline:

* **Stage 1: Preprocessing:** Input video and reference image are processed to get keypoints in the input, set up bounding boxes, obtain alignment information, etc. This stage may use components like YOLO and ViTPose. 


* **Stage 2: Inference:** Using the preprocessed input in the first stage, the diffusion/generation backbone (e.g., DiT) generates frames segment-by-segment using a sliding window strategy. This stragey employs segment concatenation and an overlapping method (to be discussed as follows). 

The preprocessing and inference happen on CPU and GPU, respectively, creating an execution pipeline. Hence, the end-to-end latency of Wan can be formulated as: $$T_{total} = T_{pre} + T_{inf}$$. We aim to optimize both stages in the pipeline without significantly impacting output quality.

---

## 2. Methods
We depict our methids in this section. 

### 2.1 Adaptive Frame Segmentation

#### 2.1.1 Problem Modeling

The second stage in Wan utilizes a sliding window strategy. In particular, Wan generates long images and videos by working in small overlapping windows instead of processing the entire canvas or timeline at once. Wan splits a large image (spatially) or a long video (temporally) into overlapping segments. The first window is generated normally. Each next window is generated while being conditioned on the previously created region, using cached features and cross‑attention to maintain continuity.The overlapping areas are blended so there are no seams or flicker.This process repeats until the full image or video is complete.

Assume that `target` and `real` are the number of frames Wan generates and the number of frames we expect to generate, respectively. `target` is often larger than `real`, resulting in computation redundancy because of the needs for staisfying concatenation divisibility and stride constraint. We formulate the relationship between `target` and `real` as follows. 
 
$$L_{target} = L_{real} + [(L_{clip} - L_{op}) - (L_{real} - L_{op}) \pmod{(L_{clip} - L_{op})}],$$

where $L_{real}$ represents for the real frame of the input video (`real_frames`), $L_{target}$ is the final number of frames needed to generate (`target_frames`). $L_{clip}$, $L_{op}$ means the length of a single clip (`clip_len`), the overlap between adjacent clips (`overlap`) respectively.

**Problems:** When `real - overlap` cannot be perfectly divided by the stride (`clip - overlap`), the input sequence is padded to the nearest divisible length, causing the generation of useless frames and a linear increase in computational overhead.

In fact, in long video generation tasks, segmenting long sequences into clips that the diffusion model can process is challenging. The traditional methods often use a fixed stride for segmentation. This segmentation strategy does not consider GPU memory capacity, and could waste edge frames when the overlapping is too much. 


#### 2.1.2 Solution (pending)

To address the above problems, we propose a reverse derivation strategy based on the constraint of GPU memory capacity.

Let $L^{\prime}_{target}$ be the new total target frames and $n$ be the number of segments. To ensure temporal continuity and maximize coverage, we establish the following constraint equation:

$$L^{\prime}_{target} = L_{clip} + (L_{clip} - L_{op}) \times n$$

Where $n \ge 0$ represents we need to perdorm $n+1$ times inference through diffusion method and is an integer. In engineering practice, we limit the upper bound of $L_{clip}$ based on Total frames of the input video, for example, $L_{clip} \le 100$. Through this method, we effectively reduce $T_{inf}$ latency.

#### 2.1.3 In-Depth Analysis
We implement the above strateg and add a knob `auto_set_lim` to allow the user to enable it. When enabling, the sytem solves the optimization problem to find the optimal $clip$ and $n$ given a $target$. Our method has the following benefits. 


* **Maximization of VRAM Utilization:** By dynamically 
Adaptively changing the clip length to maximize the utilization of GPU memory avoids padding in the traditional methods, hence avoding the waste of computation power; the GPU's tensor cores also remain high utilization at each time step. 

* **Temporal Consistency:** Our method employs a constant overlap across frames, making the video context smooth during the frame concatenation.


### 2.2 Removing Dependency for Thread-Level Parallelism

#### 2.2.1 Problem Modeling: Performance Bottleneck in Preprocessing

In the stage of preprocessing (e.g., 2D pose estimatoin and frame extraction), there is dependency between frame processing. For a video sequence containing $N$ frames, the preprocessing latency $T_{pre}$ is formulated as a cumulative sum of single-frame processing times, shown as follows.

$$T_{pre} = \sum_{i=1}^{N} (t_{decode}^{(i)} + t_{inference}^{(i)})$$

During the preprocessing, a framework is processed by one CPU core. Although we can parallelize the preprocessing by asking multiple CPU cores to work on the same frame, the thread management overhead shadows the performance benefit we can get from the parallelization. Hence, the preprocessing on CPU is slow, leading to a pipeline bubble on GPU. 

#### 2.2.2 Solution: Parallel Preprocessing

To address the above problem, we relax the depencies between frame preprocessing so that we can process multiple frames at the same time with multiple cores. In particular, we create a pool of threads (the pool size is the same as the number of CPU cores in a server).  Each CPU core is assigned with one thread, and each thread is in charge of preprocessing one frame. The parallel preprocessing of frames is possibile, because the frame preprocessing is typically implemented by a C library (e.g., OpenCV) and hence is not constrained by the Python Global Interpreter Lock (GIL). 

With the above solution, the preprocessing latency is formulated as follows: 

$$T_{pre}' \approx \frac{1}{\min(N, K)} \sum_{i=1}^{N} (t_{process}^{(i)}) + T_{overhead}$$

Where $T_{overhead}$ is the thread management overhead, and $N$ and $K$ are xxx and xxxx, respectively. 

*Note on Multi-GPU:* Multi-card setups (FSDP + ulysses_size, etc.) introduce additional communication, thread scheduling, and synchronization overhead. Reports indicate that running the original preprocessing flow in a multi-card environment actually slowed it down (up to ~170s), exhibiting typical resource contention/synchronization amplification. Therefore, additional restrictions were implemented to force preprocessing to run under single-card conditions even in multi-card environments.

#### 2.2.3 In-Depth Analysis 

* **No Limitation of GIL.** Although GIL is notoriously known for limiting multi-thread performance on CPU, our profiling shows that heavy tasks in preprocessing (e.g., `pose2d` inference and image transformations) rely heavily on underlying C/C++ kernels (such as NumPy and PyTorch operations). These operations release the GIL during preprocessing, allowing our thread pool to make the best use of CPU cores. 

* **Thread-level parallelism.** By concurrently preprocessing frames, we leverage thread-level parallelism to improve overall throughput of preprocessing. This not only maximizes the utilization of memory bandwidth, but also transforms our optimization from latency-oriented to throughput-oriented. Since the preprocessing is the performance bottleneck of the two-stage pipeline, reducing preprocessing time speeds up the whole video generation pipeline. Our method of using thread-level parallelism is very effective, reduing preprocessing time from 229s to 78s (xxxxx more details.)


---

## 3. Evaluation


* **Baseline:** Vanilla Wan (with/without SGLang).


* **Hardware A:** NVIDIA H200 (one or two cards), each of which has 141GB HBM3e.


* **Hardware B:** one AMD Instinct MI300X with 192GB HBM3.

* **Software:**

We refer to the evaluation results using our optimization techniques as Yotta-Wan in the rest of this section. 

### 3.1 Results

#### 3.1.1 End-to-End Acceleration
Table 1 shows the results collected on H200. 

**Table 1**: Evaluation results on H200
| Version | FPS | Target Frames | Preprocess (s) | Inference (s) | Total (s) | Hardware |
| --- | --- | --- | --- | --- | --- | --- |
| **Vanila** | 30 | - | 229 | 129 | 325 | 2x H200, 36 vCPU |
| **Yotta-Wan** | 30 | 153 | 79 | 133 | 212 (**↓34.8%**) | 2x H200, 36 vCPU |
| **Vanilla** | 30 | - | 229 | 232 | 684 | 1x H200, 18 vCPU |
| **Yotta-Wan** | 30 | 153 | 120 | 290 | 410 (**↓40.1%**) | 1x H200, 18 vCPU |


* **Preprocessing stage:** The time is reduced from 229s to 79s (↓65.5%), primarily due to thread-level parallelism for preprocessing and data loading.


* **Inference stage:** The time is reduced from 129s to 79s (↓38.7%).


* **Total latency:** Overall, the time is reduced by at least 34%.



#### 3.1.2 NVIDIA H200 vs. AMD MI300X (Key Comparison)

We also use Lora (lightx2v/Wan2.1-I2V-14B-720P-StepDistill-CfgDistill-Lightx2v) to improve the performance on few step (e.g. 4 steps). The method loading lora is same as Wan2.2. Actually, we simply use the original method provided in Wan2.2.

| Setup | Preprocess | Inference | Total | clip_len | target_frames | steps | GPU |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **NVIDIA Vanilla** | 94s (no replace) | 181s | 275s | 77 | 153 | 4 (no lora) | 1x H200 |
| **NVIDIA Yotta-Wan** | 93s (no replace) | 287s | 380s | 77 | 153 | 4 (lora) | 1x H200 |
| **AMD Yotta-Wan (with Lora)** | 33s (no replace) | 215s | 248s | 77 | 153 | 4 (lora) | 1x MI300X |
| **AMD Yotta-Wan (no Lora)** | 33s (no replace) | 220s | 253s | 77 | 153 | 4 (no lora) | 1x MI300X |


* **VRAM capacity advantage:** We see AMD GPU has shorter latency than NVIDIA GPU in this evaluation. This performance benefit comes from larger VRAM in AMD GPU (192 GB). This larger memory capacity is able to hold a larger $L_{clip}$, thereby reducing the segment count $n$ and total video generation time.

#### 3.1.3 Acceleration Effects with SGLang

To further validate our solution, we tested our method within the SGLang acceleration framework:

| Version | Preprocess | Inference | Total | clip_len | real/target | steps | GPU |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Origin | 224s | 472s | 696s | 77 | 155 | 20 | 1x H200 |
| Origin | 220s | 297s | 517s | 77 | 229 | 4 (no lora) | 1x H200 |
| Ours | 88s | 209s | 297s | 99 | 197 | 20 | 1x H200 |
| Ours | 90s | 159s | 249s | 99 | 197 | 4 (no lora) | 1x H200 |
| Ours | 90s | 163s | 253s | 99 | 197 | 4 (lora) | 1x H200 |

The lora method is also same as SGLang.

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
