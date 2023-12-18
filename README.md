This repos contains the code to run souporcell singlet identification step with GPU Tensorflow. It contains two helper functions to help break down the process and also make benchmarking  convenient ith different parameters.

The usage is: 

```python
used_loci_indices,used_loci_set,used_loci,loci,loci_counts,cell_counts=read_mtx(alt_matrix="alt.mtx",
                                                                                ref_matrix="ref.mtx", min_alt =5, min_ref =5 ,K=7, max_loci=1024)

clusters,posterior=cluster_step(max_loci=100,K=7,training_epochs=1000,repeats=30,cell_counts=cell_counts,
                      loci_counts=loci_counts,used_loci_indices=used_loci_indices,
                      cluster_tmp="cluster_simulated.tsv",known_cells=False,min_ref=5,min_alt=5,lr=.1)
```
