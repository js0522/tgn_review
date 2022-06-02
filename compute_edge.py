import math
import logging
import time
import sys
import argparse
import torch
import numpy as np
import pickle
from pathlib import Path

# xzl
from torch.profiler import profile, record_function, ProfilerActivity

from evaluation.evaluation import eval_edge_prediction
from model.tgn import TGN
from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder
from utils.data_processing import get_data, compute_time_statistics

# xzl
def trace_handler(p):
  output = p.key_averages().table(sort_by="self_cpu_time_total", row_limit=10)
  print(output)
  p.export_chrome_trace("./traces/trace_" + str(p.step_num) + ".json")    

torch.manual_seed(0)
np.random.seed(0)

### js) note, default -- int, str, etc.
            # action -- bool

### Argument and global variables
parser = argparse.ArgumentParser('TGN self-supervised training')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')
parser.add_argument('--bs', type=int, default=200, help='Batch_size')
parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=50, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to '
                                                                  'backprop')
parser.add_argument('--use_memory', action='store_true',
                    help='Whether to augment the model with a node memory')
parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
  "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
parser.add_argument('--message_function', type=str, default="identity", choices=[
  "mlp", "identity"], help='Type of message function')
parser.add_argument('--memory_updater', type=str, default="gru", choices=[
  "gru", "rnn"], help='Type of memory updater')
parser.add_argument('--aggregator', type=str, default="last", help='Type of message '
                                                                        'aggregator')
parser.add_argument('--memory_update_at_end', action='store_true',
                    help='Whether to update memory at the end or at the start of the batch')
parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=172, help='Dimensions of the memory for '
                                                                'each user')
parser.add_argument('--different_new_nodes', action='store_true',
                    help='Whether to use disjoint set of new nodes for train and val')
parser.add_argument('--uniform', action='store_true',
                    help='take uniform sampling from temporal neighbors')
parser.add_argument('--randomize_features', action='store_true',
                    help='Whether to randomize node features')
parser.add_argument('--use_destination_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the destination node as part of the message')
parser.add_argument('--use_source_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the source node as part of the message')
parser.add_argument('--dyrep', action='store_true',
                    help='Whether to run the dyrep model')

# xzl 
parser.add_argument('--inference_only', action='store_true',
                    help='xzl:do infer only, load a trained model')
parser.add_argument('--not_load_mem', action='store_true',
                    help='xzl:when load a trained model, not loading the memory state')
parser.add_argument('--train_split', type=float, default=0.7, help='train split. validation fixed 0.15. remaining for testing')
parser.add_argument('--mem_node_prob', type=float, default=1.0, help='%% of nodes that will have memory. default 1.0')
parser.add_argument('--fixed_edge_feature', action='store_true', default=False,
                    help="xzl:use fixed edge feature. the feature is the same as the source node's first edge")
parser.add_argument('--use_fixed_times', action='store_true', default=False,
                    help="xzl:use fixed timestamps sent to time encodings. the timestamp is cal as the avg of all ts in that batch")

try:
  args = parser.parse_args()
except:
  parser.print_help()
  sys.exit(0)

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
USE_MEMORY = args.use_memory
print(USE_MEMORY)
MESSAGE_DIM = args.message_dim   #100
MEMORY_DIM = args.memory_dim     #172
# --- below xzl ---- #
INFERENCE_ONLY = args.inference_only   
TRAIN_SPLIT = args.train_split  

Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}.pth'
get_checkpoint_path = lambda \
    epoch: f'./saved_checkpoints/{args.prefix}-{args.data}-{epoch}.pth'

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO) # xzl
Path("log/").mkdir(parents=True, exist_ok=True)
fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

### Extract data for training, validation and testing
# xzl)@new_node_val/test are nodes never showed up in training.
node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, \
new_node_test_data = get_data(DATA,
                              different_new_nodes_between_val_and_test=args.different_new_nodes, randomize_features=args.randomize_features,
                              train_split=TRAIN_SPLIT, fixed_edge_feat=args.fixed_edge_feature)

# js) NBfinder begins here 
# node centric

# Initialize training neighbor finder to retrieve temporal graph
train_ngh_finder = get_neighbor_finder(train_data, args.uniform)

# Initialize validation and test neighbor finder to retrieve temporal graph
full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

# Initialize negative samplers. Set seeds for validation and testing so negatives are the same
# across different runs
# NB: in the inductive setting, negatives are sampled only amongst other new nodes
train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
nn_val_rand_sampler = RandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations,
                                      seed=1)
test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources,
                                       new_node_test_data.destinations,
                                       seed=3)

# Set device
if GPU < 0: # xzl  much faster!! 
  print("xzl: force using cpu")
  device_string = 'cpu'  
  torch.set_num_threads(20)
  torch.set_num_interop_threads(20)
else:
  device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)

# Compute time statistics   xzl: needed by tgn model --- to normalized time diff for encoding
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
  compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

#js) mean time shift from one connection to next connection... etc... 

for i in range(args.n_runs):
  results_path = "results/{}_{}.pkl".format(args.prefix, i) if i > 0 else "results/{}.pkl".format(args.prefix)
  Path("results/").mkdir(parents=True, exist_ok=True)

# Initialize Model
      # neighbor_finder -- utils/utils/get_neighbor_finder
          # data -- utils.data_processing import get_data
          # *another way to represent the graph
      # node_feature -- pass
      # edge_feature -- pass
      # device --skipped
      # n_layers -- layer
      # n_heads -- attention layer heads
      # dropout -- dropout use in regularization method
      # use_memory --  augument model with node memory
      # message_dimension -- dimension of message
      # memory_dimension -- dimension of the memory for each user
      # memory_update_at_start -- when to update memory
      # embedding_module_type -- type of embedding modules -- GAT, graph sum, identity, time
      # message_function -- type of message function -- mlp, identity
      # aggregator_type -- type of message aggregator
      # memory_updater_type -- type of memory updater -- gru, rnn
      # n_neighbors -- number of neighbor to sample
      # mean_time_shift_src std_time -- 
      # mean_time_shift_dst std_time
            # in compute_time_statistics function
    
      # use_destination_embedding_in_message
            # whether use embedding of the destination node
            
      # use_source_embedding_in_messgae
            # whether use embedding fo the source node
      
      # dyrep
            #whether to run the DYREP model
      
      # added two
          # mem_node_prob
                #how many node have memory

          # use_fixed_times    
                #???
            
  tgn = TGN(neighbor_finder=train_ngh_finder, node_features=node_features,
            edge_features=edge_features, device=device,
            n_layers=NUM_LAYER,
            n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
            message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
            memory_update_at_start=not args.memory_update_at_end,
            embedding_module_type=args.embedding_module,
            message_function=args.message_function,
            aggregator_type=args.aggregator,
            memory_updater_type=args.memory_updater,
            n_neighbors=NUM_NEIGHBORS,
            mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
            mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
            use_destination_embedding_in_message=args.use_destination_embedding_in_message,
            use_source_embedding_in_message=args.use_source_embedding_in_message,
            dyrep=args.dyrep,
            mem_node_prob=args.mem_node_prob, use_fixed_times=args.use_fixed_times)

#loss and optim

  criterion = torch.nn.BCELoss()
  optimizer = torch.optim.Adam(tgn.parameters(), lr=LEARNING_RATE)
  
#cpu    
  tgn = tgn.to(device) # to cpu


  if not INFERENCE_ONLY: 
    # number of user node
    num_instance = len(train_data.sources) # number of instance
    # number of batch for user node
    num_batch = math.ceil(num_instance / BATCH_SIZE) # 200 per batch

    logger.info('train split: {}'.format(TRAIN_SPLIT))
    logger.info('num of training instances: {}'.format(num_instance))
    
    logger.info('num of batches per epoch: {}'.format(num_batch))
    idx_list = np.arange(num_instance)

    new_nodes_val_aps = []
    val_aps = []
    epoch_times = []
    total_epoch_times = []
    train_losses = []

    #skipped
    early_stopper = EarlyStopMonitor(max_round=args.patience)

    #start Epoch
    for epoch in range(1):
      start_epoch = time.time()
      ### Training

      # Reinitialize memory of the model at the start of each epoch
    
        #with node memory -- init to zeros
      if USE_MEMORY:
        tgn.memory.__init_memory__()

      # Train using only training graph 
      tgn.set_neighbor_finder(train_ngh_finder)
      m_loss = []

      logger.info('start {} epoch'.format(epoch))
      
    
      # start training
    
    #torch profiler
    
      with profile(
        #activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        activities=[],
        schedule=torch.profiler.schedule(
          wait=1,
          warmup=1,
          active=2),
        on_trace_ready=trace_handler,
        with_stack=True,
      ) as p:
            
        # start each batch    
            
        for k in range(0, 1, args.backprop_every): # js) 0-406
          loss = 0
          optimizer.zero_grad() # ?? to check

          # Custom loop to allow to perform backpropagation only every a certain number of batches
          for j in range(args.backprop_every):
            batch_idx = k + j
            logger.info('k {}'.format(k))
            logger.info('j {}'.format(j))
            if batch_idx >= num_batch:
              continue
            
            #               k     *     200
            # js) get the batches
            #    start/end index
            #    source node/ destination node batch
            #    edge index
            #    timestamps
            
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(num_instance, start_idx + BATCH_SIZE)
            sources_batch, destinations_batch = train_data.sources[start_idx:end_idx], \
                                                train_data.destinations[start_idx:end_idx]
            edge_idxs_batch = train_data.edge_idxs[start_idx: end_idx]
            timestamps_batch = train_data.timestamps[start_idx:end_idx]

            
            
            size = len(sources_batch)
            # xzl: how to ensure these edges are neg?  (sampled sources seem discarded)
            _, negatives_batch = train_rand_sampler.sample(size)  # js) how many sample dst nodes

            
            # js) not calc grad and init poslabel and neglabel
            with torch.no_grad():
              pos_label = torch.ones(size, dtype=torch.float, device=device)
              neg_label = torch.zeros(size, dtype=torch.float, device=device)

            tgn = tgn.train() # xzl: mark the start of training
            # xzl: @negatives_batch are neg dest. 
            #print("xzl: timestamps for the batch", timestamps_batch)
            n_samples = len(sources_batch)
            source_node_embedding, destination_node_embedding, negative_node_embedding = tgn.compute_temporal_embeddings(
      sources_batch, destinations_batch, negatives_batch, timestamps_batch, edge_idxs_batch, 10)
            #dive in