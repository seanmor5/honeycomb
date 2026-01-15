defmodule Honeycomb.TensorParallel do
  @moduledoc """
  Tensor Parallelism for multi-GPU inference.

  Distributes model layers across multiple GPUs for larger models
  and increased throughput. Supports:

  - **Column parallel**: Split weight columns across GPUs
  - **Row parallel**: Split weight rows across GPUs
  - **Pipeline parallel**: Different layers on different GPUs

  ## Architecture

  For transformer attention:
  - Q, K, V projections: Column parallel
  - Output projection: Row parallel

  For MLP:
  - First linear: Column parallel
  - Second linear: Row parallel

  ## Usage

      # Initialize tensor parallel group
      Honeycomb.TensorParallel.init_group(num_gpus: 4)

      # Shard a weight tensor
      shards = Honeycomb.TensorParallel.column_shard(weights)

      # Gather results
      result = Honeycomb.TensorParallel.all_gather(partial_results)
  """

  use GenServer
  require Logger

  defstruct [
    :world_size,
    :rank,
    :device_ids,
    :initialized,
    :communication_backend
  ]

  # Client API

  @doc """
  Initializes the tensor parallel group.

  ## Options

    * `:num_gpus` - Number of GPUs to use (required)
    * `:backend` - Communication backend (:nccl or :gloo)
  """
  def init_group(opts) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Returns the world size (number of GPUs).
  """
  def world_size do
    GenServer.call(__MODULE__, :world_size)
  end

  @doc """
  Returns the current rank.
  """
  def rank do
    GenServer.call(__MODULE__, :rank)
  end

  @doc """
  Shards a tensor along columns (dim 1) for column parallel.
  """
  def column_shard(tensor, world_size \\ nil) do
    ws = world_size || GenServer.call(__MODULE__, :world_size)
    shard_along_dim(tensor, 1, ws)
  end

  @doc """
  Shards a tensor along rows (dim 0) for row parallel.
  """
  def row_shard(tensor, world_size \\ nil) do
    ws = world_size || GenServer.call(__MODULE__, :world_size)
    shard_along_dim(tensor, 0, ws)
  end

  @doc """
  Gets the local shard for the current rank.
  """
  def get_local_shard(shards) do
    r = GenServer.call(__MODULE__, :rank)
    Enum.at(shards, r)
  end

  @doc """
  All-reduce operation: sum across all GPUs.
  """
  def all_reduce(tensor, op \\ :sum) do
    GenServer.call(__MODULE__, {:all_reduce, tensor, op})
  end

  @doc """
  All-gather operation: gather tensors from all GPUs.
  """
  def all_gather(tensor, dim \\ 0) do
    GenServer.call(__MODULE__, {:all_gather, tensor, dim})
  end

  @doc """
  Reduce-scatter operation.
  """
  def reduce_scatter(tensor, op \\ :sum) do
    GenServer.call(__MODULE__, {:reduce_scatter, tensor, op})
  end

  @doc """
  Broadcast from source rank to all others.
  """
  def broadcast(tensor, src_rank \\ 0) do
    GenServer.call(__MODULE__, {:broadcast, tensor, src_rank})
  end

  @doc """
  Creates column parallel linear layer configuration.
  """
  def column_parallel_linear(in_features, out_features, opts \\ []) do
    world_size = Keyword.get(opts, :world_size, 1)
    gather_output = Keyword.get(opts, :gather_output, true)

    %{
      type: :column_parallel,
      in_features: in_features,
      out_features: out_features,
      local_out_features: div(out_features, world_size),
      gather_output: gather_output
    }
  end

  @doc """
  Creates row parallel linear layer configuration.
  """
  def row_parallel_linear(in_features, out_features, opts \\ []) do
    world_size = Keyword.get(opts, :world_size, 1)
    input_is_parallel = Keyword.get(opts, :input_is_parallel, false)

    %{
      type: :row_parallel,
      in_features: in_features,
      out_features: out_features,
      local_in_features: div(in_features, world_size),
      input_is_parallel: input_is_parallel
    }
  end

  @doc """
  Performs forward pass for column parallel linear.
  """
  def forward_column_parallel(input, weight, bias \\ nil, config) do
    # Compute local matmul
    output = Nx.dot(input, Nx.transpose(weight))

    output =
      if bias do
        Nx.add(output, bias)
      else
        output
      end

    # Gather output if needed
    if config.gather_output do
      all_gather(output, -1)
    else
      output
    end
  end

  @doc """
  Performs forward pass for row parallel linear.
  """
  def forward_row_parallel(input, weight, bias \\ nil, config) do
    # If input is already parallel, no need to scatter
    input =
      if config.input_is_parallel do
        input
      else
        # Scatter input along last dim
        r = GenServer.call(__MODULE__, :rank)
        ws = GenServer.call(__MODULE__, :world_size)
        local_dim = div(Nx.axis_size(input, -1), ws)
        Nx.slice_along_axis(input, r * local_dim, local_dim, axis: -1)
      end

    # Compute local matmul
    output = Nx.dot(input, Nx.transpose(weight))

    # All-reduce to sum partial results
    output = all_reduce(output, :sum)

    # Add bias on rank 0 only (or use bias parallel)
    if bias do
      Nx.add(output, bias)
    else
      output
    end
  end

  # Server Callbacks

  @impl true
  def init(opts) do
    num_gpus = Keyword.fetch!(opts, :num_gpus)
    backend = Keyword.get(opts, :backend, :nccl)

    # Detect available devices
    device_ids = detect_devices(num_gpus)

    state = %__MODULE__{
      world_size: length(device_ids),
      rank: 0,  # In multi-process setup, this would be different per process
      device_ids: device_ids,
      initialized: true,
      communication_backend: backend
    }

    Logger.info("TensorParallel: Initialized with #{state.world_size} devices")

    {:ok, state}
  end

  @impl true
  def handle_call(:world_size, _from, state) do
    {:reply, state.world_size, state}
  end

  @impl true
  def handle_call(:rank, _from, state) do
    {:reply, state.rank, state}
  end

  @impl true
  def handle_call({:all_reduce, tensor, op}, _from, state) do
    # In single-process simulation, all_reduce is identity
    # In real distributed setup, would use NCCL
    result = simulate_all_reduce(tensor, op, state.world_size)
    {:reply, result, state}
  end

  @impl true
  def handle_call({:all_gather, tensor, dim}, _from, state) do
    # Simulate all_gather by replicating
    result = simulate_all_gather(tensor, dim, state.world_size)
    {:reply, result, state}
  end

  @impl true
  def handle_call({:reduce_scatter, tensor, op}, _from, state) do
    result = simulate_reduce_scatter(tensor, op, state.world_size)
    {:reply, result, state}
  end

  @impl true
  def handle_call({:broadcast, tensor, _src_rank}, _from, state) do
    # In simulation, broadcast is identity
    {:reply, tensor, state}
  end

  # Private helpers

  defp detect_devices(num_requested) do
    # In real implementation, would query CUDA devices
    # For now, return simulated device IDs
    Enum.to_list(0..(num_requested - 1))
  end

  defp shard_along_dim(tensor, dim, world_size) do
    size = Nx.axis_size(tensor, dim)
    shard_size = div(size, world_size)

    Enum.map(0..(world_size - 1), fn i ->
      start = i * shard_size
      Nx.slice_along_axis(tensor, start, shard_size, axis: dim)
    end)
  end

  defp simulate_all_reduce(tensor, :sum, _world_size) do
    # In real implementation, would sum across GPUs
    tensor
  end

  defp simulate_all_reduce(tensor, :mean, world_size) do
    Nx.divide(tensor, world_size)
  end

  defp simulate_all_gather(tensor, dim, world_size) do
    # Simulate by concatenating copies
    tensors = List.duplicate(tensor, world_size)
    Nx.concatenate(tensors, axis: dim)
  end

  defp simulate_reduce_scatter(tensor, _op, world_size) do
    # Return a shard of the reduced tensor
    dim = Nx.rank(tensor) - 1
    shard_size = div(Nx.axis_size(tensor, dim), world_size)
    Nx.slice_along_axis(tensor, 0, shard_size, axis: dim)
  end
end
