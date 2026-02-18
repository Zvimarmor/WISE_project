# Theoretical Note: Knowledge Density & Generalization

**The Relationship Between Knowledge Density and Generalization in WISE**

The efficacy of lifelong model editing is fundamentally constrained by what the authors term the "**editing dilemma**," which is rooted in the concept of **knowledge density**—defined as the average amount of information encoded per model parameter. When a model undergoes full fine-tuning or utilizes its entire parameter space for a sparse set of updates, the resulting knowledge density is low, frequently precipitating **overfitting** where the model merely memorizes query-target pairs without acquiring a structural semantic understanding. Conversely, attempting to consolidate a vast stream of edits within a common, restricted parameter manifold leads to excessively high density, which triggers **catastrophic forgetting** and internal knowledge conflicts.

To navigate this trade-off, WISE introduces a knowledge-sharding mechanism that partitions the edit stream into $k$ distinct shards, each assigned to a random parameter subspace defined by a binary gradient mask $M_i$ with a sparsity ratio $\rho$. By constraining updates to these subspaces such that $k \cdot \rho < 1$, the framework artificially increases the knowledge density within the active manifold, thereby regularizing the optimization process and compelling the model to learn more robust, generalizable features. 
However, this density has a finite ceiling; when a subspace reaches "mask memory exhaustion," further edits lead to significant interference. To mitigate this, WISE utilizes the sub-orthogonality of these random masks to merge the $k$ shards into a unified side memory using Ties-Merge, leveraging the calculated $\rho^k$ overlap as "anchors" to synchronize model behavior across subspaces while resolving directional conflicts.

### Capacity Management and Dynamic Retrieval in WISE

While a single side memory provides an efficient space for updates, it has a limited knowledge capacity. According to the researchers, a state of "**mask memory exhaustion**" occurs when the parameters within the allocated subspaces (e.g., the 20% of weights defined by $\rho=0.2$) can no longer accommodate new edits without causing interference or performance degradation. Empirically, the sources suggest that these subspaces can reliably store at least 500 edited samples before reaching this threshold.

When the system detects that the current side memory is full, meaning all $k$ subspaces have been utilized and merged, it transitions from a merging strategy (**WISE-Merge**) to a retrieval strategy (**WISE-Retrieve**). Instead of continuing to compress new information into the same overloaded matrix, the algorithm allocates a new side memory container (a fresh copy of the targeted FFN layer's weights). This allows the model to scale to thousands of edits by maintaining an ensemble of "expert" side memories.

During inference, the challenge is determining which side memory contains the relevant knowledge for a given query. WISE solves this using an **activation-based routing mechanism**. For every input, the model calculates an **activation indicator score** ($\Delta act$) for each side memory in the pool. This score measures the L2 norm of the difference between the side memory’s output and the main memory’s output:

$$
\Delta act(x)=\|A(x)\cdot(W'_v - W_v)\|_2
$$

The algorithm then selects the side memory that yields the maximal activation score. Intuitively, the memory that reacts most strongly to the input is identified as the one containing the specific "sharded" knowledge required to answer the query, while irrelevant queries naturally produce low scores across all side memories, routing the model back to its original pretrained knowledge.
