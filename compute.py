def compute_time_statistics(sources, destinations, timestamps):
  last_timestamp_sources = dict()
  last_timestamp_dst = dict()
  interval_sources = dict()  #xzl
  src_edge_cnt = dict() # xzl
  all_timediffs_src = []
  all_timediffs_dst = []
  for k in range(len(sources)):
    source_id = sources[k]
    dest_id = destinations[k]
    c_timestamp = timestamps[k]
    if source_id not in last_timestamp_sources.keys():
      last_timestamp_sources[source_id] = 0
      src_edge_cnt[source_id] = 1 # xzl
    else:
      src_edge_cnt[source_id] += 1 # xzl      
    if dest_id not in last_timestamp_dst.keys():
      last_timestamp_dst[dest_id] = 0
    # xzl
    if last_timestamp_sources[source_id] != 0:  # interacted before 
      if source_id not in interval_sources.keys():
        interval_sources[source_id] = []
      interval_sources[source_id].append(c_timestamp - last_timestamp_sources[source_id])
    all_timediffs_src.append(c_timestamp - last_timestamp_sources[source_id])
    all_timediffs_dst.append(c_timestamp - last_timestamp_dst[dest_id])
    last_timestamp_sources[source_id] = c_timestamp
    last_timestamp_dst[dest_id] = c_timestamp
  '''
  # xzl: deal with src with only 1 interaction.... no good solution. we are done, last ts?   
  c_timestamp = timestamps[-1]
  for src, ts in last_timestamp_sources.items():
    #assert (ts != 0)
    #assert src not in interval_sources.keys()
    if src not in interval_sources.keys():
      interval_sources[src] = []
    interval_sources[src].append(c_timestamp-ts)
    #interval_sources[src] = [float("inf")] # only interaction once. max
  '''

  print_stat = False
  if print_stat:
    print("--- xzl ---")
    #interval_medians = dict()
    interval_medians = []
    intervals = []
    for s,i in interval_sources.items():
      #interval_medians[s] = np.median(i)
      interval_medians.append(np.median(i))
      intervals += i
    edge_cnt = []
    #for s,i in interval_sources.items():
    #  interval_cnt.append(len(i))
    for s,i in src_edge_cnt.items():
      edge_cnt.append(i)
    hist = np.histogram(edge_cnt, bins=20)
    print('hist for per-node edge cnt', hist)
    #print(np.histogram(edge_cnt, bins=[0,1,2,3,4,5,10,100,1000,10000]))
    plt.hist(edge_cnt, bins=20)
    plt.savefig("hist-edge-cnt.png")
      
    hist2 = np.histogram(intervals, bins=20)
    print("intervals for all nodes", hist2)

    hist2 = np.histogram(interval_medians, bins=20)
    print("median intervals for all nodes", hist2)
    
    mask = [x < hist[1][1] for x in src_edge_cnt]
    #print(mask)
    masked_medians=np.array(interval_medians)[mask]
    hist2 = np.histogram(masked_medians, bins=20)
    print("median intervals for nodes w/ fewer edges", hist2)

  assert len(all_timediffs_src) == len(sources)
  assert len(all_timediffs_dst) == len(sources)
  mean_time_shift_src = np.mean(all_timediffs_src)
  std_time_shift_src = np.std(all_timediffs_src)
  mean_time_shift_dst = np.mean(all_timediffs_dst)
  std_time_shift_dst = np.std(all_timediffs_dst)

  return mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst