# Code Exploration: WISE Implementation

This document details the optional code investigation into the WISE method's core algorithms: subspace allocation, WISE-Merge, and Ties-Merge.

## 1. WISE-Merge Algorithm

**Location**: `easyeditor/models/wise/WISE.py` -> `WISEAdapter.merge_weight()`

The WISE-Merge algorithm (specifically the `wise_merge` variant) accumulates multiple edits in a "memory" buffer (`memory_weight`) and then merges them into the permanent model weights at a specific frequency (`merge_freq`).
The merging technique is according to: TIES-Merging: Resolving Interference When Merging Models, Yadav et al. 2023, [https://doi.org/10.48550/arXiv.2306.01708](https://doi.org/10.48550/arXiv.2306.01708).

### Pseudocode

```python
def merge_weight(self):
    """
    Merges accumulated edit weights into the permanent model layer.
    """
    # 1. Check if merge frequency is met (handled by caller in main loop)
    # self.config.merge_freq // self.config.save_freq determines batch size
  
    # 2. Select the merging algorithm (e.g., 'ties')
    merge_alg = self.merge_dict[self.config.merge_alg] 
  
    # 3. Prepare weights and densities for the merge
    # Assign equal weight to each memory shard being merged
    num_shards = len(self.memory_weight)
    weights = [self.config.weights / num_shards] * num_shards
    densities = self.config.densities
  
    # 4. Execute the Merge (using Ties-Merge / GTA)
    # This combines the base layer with the accumulated memory updates
    merged_delta = merge_alg.execute(
        weights=weights,
        base=self.original_layer.weight,
        tensors=self.memory_weight,  # The list of K subspace updates
        densities=densities
    )
  
    # 5. Update the Model
    # The result becomes the new "permanent" weight for this side memory
    self.memory_weight = [merged_merge_weight] 
    # (Note: In some config branches, it clears memory; in others (retrieve), it keeps the merged result as a new base)
```

---

## 2. Subspace Allocation (New Edit Data)

**Location**: `easyeditor/models/wise/WISE.py` -> `WISEAdapter.generate_non_overlapping_mask()`

WISE ensures that each new edit operates in a distinct subspace by generating non-overlapping masks. This prevents interference between sequential edits before they are merged.

### How it works:

1. **Tracking Usage**: The class maintains a boolean array `self.used_mask` that tracks which parameters (indices) have already been assigned to a subspace.
2. **Availability Check**: It identifies all `available_indices` where `used_mask` is `False`.
3. **Random Sampling**: It randomly samples `mask_size` indices from these available spots. `mask_size` is determined by `mask_ratio` (e.g., 0.2% of total params).
4. **Update**: It marks these new indices as `True` in `used_mask` so they won't be picked for the next edit.
5. **Assignment**: The gradient mask `self.weight_mask` is set to 1 at these indices, forcing the optimizer to only update these specific parameters.

```python
def generate_non_overlapping_mask(self, mask_ratio):
    # 1. Calculate how many params to use
    total_params = self.new_weight.numel()
    mask_size = int(mask_ratio * total_params)

    # 2. Initialize usage tracker if needed
    if self.used_mask is None:
        self.used_mask = zeros(total_params, dtype=bool)

    # 3. Find available (unused) parameters
    available_indices = where(~self.used_mask)[0]

    # 4. Error if full ("Mask Memory Exhaustion")
    if len(available_indices) < mask_size:
        raise ValueError("Memory Exhaustion: No parameters left for new subspace.")

    # 5. Randomly select new indices
    chosen_indices = random.choice(available_indices, size=mask_size, replace=False)

    # 6. Create the mask and marking as used
    mask_array = zeros(total_params)
    mask_array[chosen_indices] = 1
    self.used_mask[chosen_indices] = True 
  
    self.weight_mask = to_tensor(mask_array)
```

---

## 3. Ties-Merge Implementation

**Location**: `easyeditor/models/wise/merge/gta.py` -> `GTA.execute()` (Generalized Task Arithmetic)

WISE uses **Ties-Merge** (referenced as `ties` in `merge_dict` and implemented via `GTA` class) to resolve conflicts when merging the $k$ subspaces.

### Mechanism

The `GTA.execute` function performs three key steps (Trim, Elect, Merge):

1. **Task Vectors**: Calculates the change ($\Delta$) for each subspace: `delta = memory_weight_i - base_weight`.
2. **Sparsify (Trim)**:
   * It uses `magnitude` sparsification (filtering out small changes).
   * Keeps only the top parameters based on `densities`.
3. **Sign Consensus (Elect)**:
   * This is the critical "Resolve Conflicts" step.
   * It sums the signs of all deltas for each parameter (`sign_weight = delta.sum(dim=0)`).
   * **Majority Rule**: If the sum is $\ge 0$, the "elected" sign is +1; otherwise -1.
   * **Filtering**: Any individual delta that *disagrees* with the majority sign for a specific parameter is masked out (zeroed). This prevents one subspace from pulling a parameter positive while another pulls it negative (interference).
4. **Aggregate (Merge)**:
   * It takes the weighted average of the surviving (sign-consistent) deltas.
   * `mixed_delta = (weighted_deltas * mask).sum() / divisor`
5. **Final Update**:
   * `final_weight = base + mixed_delta`

### Libraries Used

* **PyTorch (`torch`)**: For all tensor operations (sums, masks, linear algebra).
* **NumPy (`numpy`)**: Used in `WISE.py` for random sampling of indices, but `gta.py` relies primarily on pure PyTorch functions.

---

## 4. Deep Dive: Masking, Insertion, and Routing

### A. Creating and Masking the $M_i$ Matrix

**Location**: `easyeditor/models/wise/WISE.py`

The mask $M_i$ (binary gradient mask) is created to define the "subspace" for the current edit batch.

* **Creation**: The method `generate_non_overlapping_mask` (Lines 483-496) creates the binary mask.
  * It selects random indices that are *not* currently used (`~self.used_mask`).
  * It sets `mask_array[chosen_indices] = 1`.
  * This `mask_array` becomes `self.weight_mask`.

```python
# WISE.py: Lines 483-496
def generate_non_overlapping_mask(self, mask_ratio):
    # ... calculates mask_size ...
    chosen_indices = np.random.choice(available_indices, size=mask_size, replace=False)
    mask_array = np.zeros(p_grad.size()[0], dtype=int)
    mask_array[chosen_indices] = 1
    self.weight_mask = torch.from_numpy(mask_array).to(p_grad.device) # The Matrix M_i
```

### B. Inserting New Information into Subspaces

**Location**: `easyeditor/models/wise/WISE.py` -> `edit()` and `mask_new_weight_gradient()`

The "insertion" happens during the optimization (training) loop. The algorithm forces the *update* to stay within the subspace defined by $M_i$.

* **Process**:
  1. The standard backpropagation calculates gradients for *all* weights (`loss.backward()`, Line 212).
  2. **Crucially**, it immediately calls `self.get_adapter_layer().mask_new_weight_gradient()` (Line 213).
  3. Inside that function (Lines 500-508), the gradient is multiplied by the mask: `p_grad = p_grad * self.weight_mask`.
  4. This zeroes out any gradient updates *outside* the chosen subspace, ensuring information is only "written" to the intended shards.

#### Decision Logic: When is a Subspace "Full"?
The decision to switch to a new subspace is controlled by the **`save_freq`** hyperparameter, not by a dynamic density check.

* **Rotation Trigger**: In `WISE.py` (Lines 237-239), there is a counter `editing_total_cnt`.
    ```python
    if self.config.save_freq is not None and editing_total_cnt % self.config.save_freq == 0:
        self.get_adapter_layer().save_weight() # Saves current subspace to memory
        # ... logic to create new mask for next batch ...
    ```
    **Verified Value**: across all provided configuration files (e.g., `llama-7b.yaml`, `gpt2-xl.yaml`, `mistral-7b.yaml`), this value is consistently set to **`save_freq: 500`**. This means the model creates a new side memory shard every 500 edits.

* **"Fullness" & Exhaustion**:
  * The `generate_non_overlapping_mask` function (if used) tracks utilization via `self.used_mask`.
  * It checks `available_indices`. If `len(available_indices) < mask_size`, it raises a `ValueError` (Memory Exhaustion).
  * **In Summary**: The algorithm "rotates" to a new subspace every `save_freq` edits. It explicitly checks if "the subspace is full" (i.e., NO more params available to allocate) only when generating the mask, throwing an error if the model runs out of capacity.

### C. Subspace Selection & Routing (The "How it Knows")

#### 1. During Editing (Training)

How does it know which subspace to use?

* It doesn't "select" from a list during the edit process itself. Instead, it **creates/assigns** the current active subspace at the start of the edit batch (or periodically using `save_freq`).
* The `WISEAdapter` has a single `new_weight` buffer it creates.
* Once a subspace is filled or the `save_freq` is hit (Lines 150-151), it generates a *new* mask. The optimizer then writes to this new masked area.
* Essentially: **The "Active Subspace" is simply the current `weight_mask` applied to the current `new_weight` buffer.**

#### 2. During Inference (Routing to Side Memory)

**Location**: `easyeditor/models/wise/WISE.py` -> `WISEAdapter.forward()` (Lines 510-550)

This is where the dynamic routing happens (WISE-Retrieve).

* **The Logic**:
  * The model maintains a list of expert side memories: `self.memory_weight`.
  * For an input `x`, it calculates the output of the Main Memory (`original_layer_output`) and *each* Side Memory (`memory_weight_layer_output`).
  * It computes the **Activation Score** ($\Delta act$) using the `euc` function (Euclidean distance/L2 norm) on **Line 546**:
    `dist = euc(original_layer_output, memory_weight_layer_output, ...)`
  * **Selection**: It iterates through all memories and picks the one with the highest `dist` (Line 547-549):
    ```python
    if dist > min_dist and dist > threshold:
         layer_out = memory_weight_layer_output # Route to this memory
         min_dist = dist
    ```
  * It effectively says: "Which side memory causes the biggest change in activation compared to the original model? Use that one."
