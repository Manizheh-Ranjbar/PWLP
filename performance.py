import json
import os

from torch_geometric.profile import get_stats_summary, get_model_size, count_parameters, get_data_size, \
    get_cpu_memory_from_gc, get_gpu_memory_from_gc
from torch_geometric.profile.utils import byte_to_megabyte


def profile_helper(all_stats, model, train_dataset, stats_suffix):
    summarized_stats = get_stats_summary(all_stats)
    model_size = get_model_size(model)
    parameters = count_parameters(model)
    train_dataset_size = get_data_size(train_dataset.data)
    cpu_usage = get_cpu_memory_from_gc()
    gpu_usage = get_gpu_memory_from_gc()
    stats = {}
    print("------------------------------------------")

    print(f"Summarized stats: {summarized_stats}")
    stats[
        'Average Time(in seconds)'
    ] = f'{summarized_stats.time_mean:.2f} ± {summarized_stats.time_std:.2f}'

    # Details about there params are here: https://pytorch.org/docs/stable/generated/torch.cuda.memory_stats.html
    # we use all: combined statistics across all memory pools and peak: maximum value of this metric
    # for all below metrics and convert to megabytes

    # Returns the max of current GPU memory occupied by tensors in bytes for a given device.
    stats['Max Allocated CUDA (in MegaBytes)'] = f'{summarized_stats.max_allocated_cuda:.2f}'
    # Returns the max of current GPU memory managed by the caching allocator in bytes for a given device
    stats['Max Reserved CUDA (in MegaBytes)'] = f'{summarized_stats.max_reserved_cuda:.2f}'
    # amount of active memory in bytes.
    stats['Max Active CUDA (in MegaBytes)'] = f'{summarized_stats.max_active_cuda:.2f}'

    # from command: 'nvidia-smi --query-gpu=memory.free --format=csv'
    stats['Min NVIDIA SMI Free CUDA Memory (in MegaBytes)'] = f'{summarized_stats.min_nvidia_smi_free_cuda}'
    # from command: 'nvidia-smi --query-gpu=memory.used --format=csv'
    stats['Max NVIDIA SMI Used CUDA Memory (in MegaBytes)'] = f'{summarized_stats.max_nvidia_smi_used_cuda}'

    print("------------------------------------------")

    print(f"Model size: {model_size}")
    stats['Model size (in MegaBytes)'] = f'{byte_to_megabyte(model_size):.2f}'
    print(f"Parameters: {parameters}")
    stats['Number of Model Parameters'] = f'{parameters}'
    print(f"Train Dataset Size: {train_dataset_size}")
    stats['Train Dataset Size (in MegaByes)'] = f'{byte_to_megabyte(model_size):.2f}'

    print("------------------------------------------")

    print(f"CPU usage: {cpu_usage}")
    print(f"GPU usage: {gpu_usage}")

    print("------------------------------------------")
    os.makedirs('stats', exist_ok=True)
    with open(f'stats/stats_{stats_suffix}.json', 'w') as stats_file:
        json.dump(stats, stats_file)
    print("fin profiling.")

################################

max_memory_allocated = torch.cuda.max_memory_allocated()
print(f"Max allocated CUDA memory: {max_memory_allocated / (1024 ** 2)} MB")

model_size_params = sum(p.numel() for p in classifier.parameters())
print(f"NUM parameters: {model_size_params} parameters")

# Calculate the size of the model in bytes
model_size_bytes = model_size_params * 4  # 4 bytes per float32 parameter

# Convert size to megabytes (1 MB = 1024 * 1024 bytes)
model_size_mb = model_size_bytes / (1024 ** 2)

print(f"NUM parameters: {model_size_params / 1e6:.3f}M parameters")
print(f"Model size: {model_size_mb:.2f} MB")


with open(args.result_dir+'/'+args.data_name+'cuda_size_param_results'+'.txt', 'a+') as f:
    f.write(str(max_memory_allocated / (1024 ** 2)) + '\n')
    f.write(str(model_size_mb) + '\n')
    f.write(str(model_size_params / 1e6) + '\n')
    f.write(str(end - start) + '\n')
    f.write(str(end_training - start_training) + '\n')
#//

parameters = list(classifier.parameters())
# emb = torch.nn.Embedding(net[0], cmd_args.hidden).to(args.device)
# torch.nn.init.xavier_uniform_(emb.weight)
# parameters += list(emb.parameters())

# if args.train_node_embedding:
#     torch.nn.init.xavier_uniform_(emb.weight)
    # parameters += list(emb.parameters())
# total_params = sum(p.numel() for param in parameters for p in param)
total_params = sum(p.numel() for p in classifier.parameters())
# total_params+=list(emb.parameters())
print(f'Total number of parameters like scaled is {total_params}')

##########################
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")


total_size = 0
for param_state in optimizer.state.values():
    for key, value in param_state.items():
        if isinstance(value, torch.Tensor):  # Ensure it's a tensor
            total_size += value.numel() * value.element_size()  # Compute size in bytes
Optimizer_size_mb = total_size / 1024**2
print(f"Optimizer state size: {Optimizer_size_mb:.2f} MB")  # Convert to MB
print(f"Model size: {model_size_mb:.2f} MB\n")
print(f"total size: {Optimizer_size_mb+model_size_mb:.2f} MB\n")

#####################

with open(args.result_dir+'/'+args.data_name+'cuda_size_param_results_details'+'.txt', 'a+') as f:
    f.write(f"Max allocated CUDA memory: {max_memory_allocated / (1024 ** 2)} MB" + '\n')
    f.write(f"Max Reserved CUDA memory: {torch.cuda.memory_reserved() / (1024 ** 2)} MB" + '\n')
    f.write(f"NUM parameters: {model_size_params} parameters\n")
    f.write(f"NUM parameters: {model_size_params / 1e6:.3f}M parameters\n")
    f.write(f"Model size: {model_size_mb:.2f} MB\n")
    f.write(f'Total number of parameters like scaled is {total_params}\n')
    f.write(f"Time taken for run: {end - start:.2f} seconds\n")
    f.write(f"Time taken for run: {end_training - start_training:.2f} seconds\n")
    
    f.write(f"Optimizer state size: {Optimizer_size_mb:.2f} MB\n")  # Convert to MB
    f.write(f"Model size: {model_size_mb:.2f} MB\n")
    f.write(f"total  model_optimizer size: {Optimizer_size_mb+model_size_mb:.2f} MB\n")


print("**********************")
print(torch.cuda.memory_summary())



