def read_mtx(min_alt = 5,min_ref = 5,max_loci =1024,alt_matrix="alt_simulated.mtx",ref_matrix="ref_simulated.mtx"):
    '''
    input:
        - alt_matrix: the alternative loci matrix obtained from vartrix or similar program
        - ref_matrix: the reference loci matrix  obtained from vartrix or similar program
        -  min_alt: minimum number of cells covering an alternative loci in order to consider this variant
        -  min_ref: minimum number of cells covering a reference loci in order to consider this variant
        - max_loci: maximum number of loci used (default = 1024)
    ''' 
    cell_index = {}
    total_lost = 0
    loci_counts = {}
    cell_counts = {}
    with open(alt_matrix) as alt:
        alt.readline()
        alt.readline()
        tokens = alt.readline().strip().split()
        cells = int(tokens[1])
        cell_counts = {x: {} for x in range(1, cells + 1)}
        total_loci = int(tokens[0])
        for line in alt:
            tokens = line.strip().split()
            locus = int(tokens[0])
            cell = int(tokens[1])
            cell_counts.setdefault(cell, {})
            count = int(tokens[2])
            cell_counts[cell][locus] = [0, count]
            loci_counts.setdefault(locus, [0, 0])
            if count > 0:
                loci_counts[locus][1] += 1
    with open(ref_matrix) as alt:
        alt.readline()
        alt.readline()
        alt.readline()
        for line in alt:
            tokens = line.strip().split()
            locus = int(tokens[0])
            cell = int(tokens[1])
            count = int(tokens[2])
            cell_counts[cell][locus][0] = count
            loci_counts.setdefault(locus, [0, 0])
            if count > 0:
                loci_counts[locus][0] += 1

    used_loci_set = set()
    used_loci = []
    for (locus, counts) in loci_counts.items():
        if counts[0] >= min_ref and counts[1] >= min_alt:
            used_loci.append(locus - 1)
            used_loci_set.add(locus - 1)
    used_loci = sorted(used_loci)
    used_loci_indices = {locus:i for (i, locus) in enumerate(used_loci)}
    loci = len(used_loci)
    return used_loci_indices,used_loci_set,used_loci,loci,loci_counts,cell_counts


def cluster_step(max_loci,K,training_epochs,repeats,cell_counts,loci_counts,used_loci_indices,  known_cells=False,min_ref=5,min_alt=5,lr=.1):
    '''
        input: 
            - max_loci: maximum number of loci used, this should be similar to the one given in previous step
            - K: number of clusters
            - training_epochs: number of epochs for Adam optimizer. default = 100
            - repeats: number of repeats for the algorithm. default = 15
            - cell_counts: the cell_counts dict obtained from the previous step
            - loci_counts: the loci_counts dict obtained from the previous step 
            - used_loci_indices: the loci indices used generated prom the previous step
        output:
            - a list of [cluster assignment, posterior] from the best run 
    '''
    print("loci being us based on min_alt, min_ref, and max_loci "+str(loci))
    
    cells = len(cell_counts)
    total_lost = 0

    cell_data = np.zeros((cells, max_loci))
    cell_loci = np.zeros((cells, max_loci))
    
    weights = np.zeros((cells, max_loci))
    
    for cell in cell_counts.keys():
        index = 0
        single_cell_counts = cell_counts[cell]
        
        #prioritize locus that is highly expressed across cells:
        #for this given cell, get the set of 
        
        this_cell_locus=list(cell_counts[cell].keys())
       
             
        for locus in this_cell_locus:
            locus_counts = single_cell_counts[locus]
            if loci_counts[locus][0] >= min_ref and loci_counts[locus][1] >= min_alt:
                if index < max_loci:
                    ref_c = locus_counts[0]
                    alt_c = locus_counts[1]
                    if ref_c + alt_c > 0:
                        cell_data[cell - 1][index] = float(ref_c)/float(ref_c + alt_c) if ref_c + alt_c > 0 else 0.0
                        cell_loci[cell - 1][index] = used_loci_indices[locus - 1]
                        weights[cell - 1][index] = 1.0
                        index += 1
                    total_lost += 1
    ### set 0 weights for cells from other clusters:
    random_size=100


    data = cell_data
    data_loci = cell_loci
    weightshape_np=np.array(np.transpose(np.broadcast_to(weights.T,(K,max_loci,weights.shape[0]))))

   
        
    if known_cells  is not None:
        assert len(known_cells.keys()) <= K
        assert known_cells.columns==["index","cluster"]
        #did a quick check for unique assignment to a cluster:
        freq=known_cell["cluster"].value_counts()
        assert np.max(freq)==1

        for i,j in enumerate(known_cells.keys()):
            for k in range(K):
                if k!=i:
                    weightshape_np[known_cells[j]-1,:,k]=weightshape_np[known_cells[j]-1,:,k]*10000    
    
    cells = data.shape[0]

    #save this for investigation:
    print("save cell_data and cell_loci:")
     
    rng = np.random
    
    
    import tensorflow as tf
    
    session_conf = tf.compat.v1.ConfigProto(
          allow_soft_placement=True)
    session_conf.gpu_options.allow_growth = True
    session_conf.gpu_options.per_process_gpu_memory_fraction =1
    tf.compat.v1.disable_eager_execution()
    
    tf.compat.v1.reset_default_graph()
    
    with tf.device("/GPU:0"):
    

        #init = tf.compat.v1.constant(sample_genotypes.T)
        #phi = tf.compat.v1.get_variable(name="phi", initializer = init, dtype = tf.float64)
        
        phi = tf.compat.v1.get_variable(name = "phi",shape=(loci, K), initializer = tf.initializers.random_uniform(minval = 0, maxval = 1), dtype = tf.float64)

        input_data = tf.compat.v1.placeholder("float64", (cells, max_loci)) #tf.constant("input",np.asmatrix(data))
        input_loci = tf.compat.v1.placeholder("int32", (cells, max_loci))
        loci_per_cell = tf.compat.v1.placeholder("float64", (cells))
        trans = tf.compat.v1.transpose(input_data)
        broad_trans = tf.compat.v1.broadcast_to(trans,[K,max_loci,cells])
        untrans = tf.compat.v1.transpose(broad_trans)
        xtest = untrans-tf.compat.v1.gather(phi,input_loci)
        weight_data = tf.compat.v1.placeholder("float64", (cells,max_loci,K)) #tf.constant("weights",np.asmatrix(weights))
    
        weighted = weight_data*xtest
        powtest = -tf.compat.v1.pow(weighted,2)
        post = tf.compat.v1.reduce_sum(powtest,axis=1)
        logsum = tf.compat.v1.reduce_logsumexp(post,axis=1)
        cost = -tf.compat.v1.reduce_sum(logsum)
    
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr).minimize(cost)
            
        posteriors = []
        min_cost = None
        weightsshape=[]
        logsum_list=[]
        cluster_list=[]
    for repeat in range(repeats):
        init = tf.compat.v1.global_variables_initializer()
        print("repeat "+str(repeat))
        training_epochs = 1000
        last_cost = None
        with tf.compat.v1.Session(config = session_conf) as sess:
            sess.run(init)
            for epoch in range(training_epochs):
                sess.run(optimizer, feed_dict={input_data:data, weight_data:weightshape_np, input_loci:data_loci})
    
                if epoch % 10 == 0:
                    c = sess.run(cost, feed_dict={input_data:data, weight_data:weightshape_np, input_loci:data_loci})
                    print("epoch "+str(epoch)+" "+str(c))
                    #if last_cost and ((last_cost-c)/c) < 0.0001:
                    if min_cost and last_cost and c > min_cost and (last_cost - c)/(c - min_cost) < 0.005:
                        print("bailing out, too little progress toward minimum so far")
                        break
                    if last_cost and last_cost - c < 1:
                        last_cost = None
                        break
                    last_cost = c
            if min_cost:
                min_cost = min(min_cost, c)
            else:
                min_cost = c
    
            posterior = sess.run(post, feed_dict={input_data:data, weight_data:weightshape_np, input_loci:data_loci})
            posteriors.append((c,posterior))
                
                
    
    sess.close()
    
    posterior = sorted(posteriors, key=lambda x: x[0])
    posterior = posterior[0][1]    
    clusters = np.argmax(posterior,axis=1)
    return(list(clusters,posterior))