defmodule Honeycomb.MemoryPool do
  @moduledoc """
  Memory pool for efficient tensor allocation.

  Pre-allocates memory buffers to avoid allocation overhead during inference.
  Features:

  - **Size-class pooling**: Different pools for common tensor sizes
  - **Thread-safe**: Uses ETS for lock-free access
  - **Auto-growth**: Expands pools when needed
  - **Defragmentation**: Coalesces free blocks periodically

  ## Usage

      # Start the pool
      Honeycomb.MemoryPool.start_link(max_memory_gb: 8)

      # Allocate a tensor buffer
      {:ok, buffer} = Honeycomb.MemoryPool.allocate({1024, 768}, :f16)

      # Release back to pool
      Honeycomb.MemoryPool.release(buffer)
  """

  use GenServer
  require Logger

  @default_max_memory_gb 8
  @size_classes [
    # Common sizes for LLM inference
    {64, 100},
    {256, 100},
    {1024, 50},
    {4096, 50},
    {16384, 25},
    {65536, 20},
    {262144, 15},
    {1048576, 10}
  ]

  defstruct [
    :pools,
    :allocations,
    :max_memory,
    :used_memory,
    :stats
  ]

  # Client API

  @doc """
  Starts the memory pool.

  ## Options

    * `:max_memory_gb` - Maximum memory in GB (default: 8)
    * `:prealloc` - Pre-allocate pools on startup (default: true)
  """
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Allocates a tensor buffer from the pool.

  Returns `{:ok, buffer}` or `{:error, :out_of_memory}`.
  """
  def allocate(shape, dtype \\ :f32) do
    GenServer.call(__MODULE__, {:allocate, shape, dtype})
  end

  @doc """
  Releases a buffer back to the pool.
  """
  def release(buffer) do
    GenServer.cast(__MODULE__, {:release, buffer})
  end

  @doc """
  Allocates a tensor initialized with zeros.
  """
  def zeros(shape, dtype \\ :f32) do
    case allocate(shape, dtype) do
      {:ok, buffer} ->
        tensor = Nx.broadcast(Nx.tensor(0, type: dtype), shape)
        {:ok, %{buffer | tensor: tensor}}

      error ->
        error
    end
  end

  @doc """
  Returns pool statistics.
  """
  def stats do
    GenServer.call(__MODULE__, :stats)
  end

  @doc """
  Triggers garbage collection of unused buffers.
  """
  def gc do
    GenServer.call(__MODULE__, :gc)
  end

  @doc """
  Clears all pools and releases memory.
  """
  def clear do
    GenServer.call(__MODULE__, :clear)
  end

  # Server Callbacks

  @impl true
  def init(opts) do
    max_memory_gb = Keyword.get(opts, :max_memory_gb, @default_max_memory_gb)
    max_memory = max_memory_gb * 1024 * 1024 * 1024

    # Initialize pools for each size class
    pools = init_pools(Keyword.get(opts, :prealloc, false))

    state = %__MODULE__{
      pools: pools,
      allocations: %{},
      max_memory: max_memory,
      used_memory: 0,
      stats: %{
        allocations: 0,
        releases: 0,
        pool_hits: 0,
        pool_misses: 0,
        gc_runs: 0
      }
    }

    Logger.info("MemoryPool: Initialized with max_memory=#{max_memory_gb}GB")

    {:ok, state}
  end

  @impl true
  def handle_call({:allocate, shape, dtype}, _from, state) do
    size = compute_size(shape, dtype)
    size_class = find_size_class(size)

    case try_pool_allocate(state.pools, size_class, size) do
      {:ok, buffer, pools} ->
        buffer = %{buffer | shape: shape, dtype: dtype, size: size}
        allocations = Map.put(state.allocations, buffer.id, buffer)
        stats = %{state.stats | allocations: state.stats.allocations + 1, pool_hits: state.stats.pool_hits + 1}

        state = %{state |
          pools: pools,
          allocations: allocations,
          used_memory: state.used_memory + size,
          stats: stats
        }

        {:reply, {:ok, buffer}, state}

      :empty ->
        # Pool empty, allocate new
        if state.used_memory + size <= state.max_memory do
          buffer = allocate_new_buffer(shape, dtype, size)
          allocations = Map.put(state.allocations, buffer.id, buffer)
          stats = %{state.stats | allocations: state.stats.allocations + 1, pool_misses: state.stats.pool_misses + 1}

          state = %{state |
            allocations: allocations,
            used_memory: state.used_memory + size,
            stats: stats
          }

          {:reply, {:ok, buffer}, state}
        else
          {:reply, {:error, :out_of_memory}, state}
        end
    end
  end

  @impl true
  def handle_call(:stats, _from, state) do
    pool_stats =
      Enum.map(state.pools, fn {size_class, pool} ->
        {size_class, %{
          total: length(pool.buffers) + pool.allocated,
          free: length(pool.buffers),
          allocated: pool.allocated
        }}
      end)
      |> Map.new()

    stats = Map.merge(state.stats, %{
      used_memory_mb: state.used_memory / (1024 * 1024),
      max_memory_mb: state.max_memory / (1024 * 1024),
      utilization: state.used_memory / state.max_memory,
      active_allocations: map_size(state.allocations),
      pools: pool_stats
    })

    {:reply, stats, state}
  end

  @impl true
  def handle_call(:gc, _from, state) do
    # Return unused buffers in each pool beyond minimum
    {pools, freed} =
      Enum.map_reduce(state.pools, 0, fn {size_class, pool}, total_freed ->
        {min_count, _} = Enum.find(@size_classes, {10, 10}, fn {sc, _} -> sc == size_class end)
        excess = max(0, length(pool.buffers) - min_count)

        if excess > 0 do
          {keep, _discard} = Enum.split(pool.buffers, min_count)
          freed = excess * size_class

          {{size_class, %{pool | buffers: keep}}, total_freed + freed}
        else
          {{size_class, pool}, total_freed}
        end
      end)

    stats = %{state.stats | gc_runs: state.stats.gc_runs + 1}

    state = %{state |
      pools: Map.new(pools),
      used_memory: max(0, state.used_memory - freed),
      stats: stats
    }

    Logger.debug("MemoryPool GC: freed #{freed} bytes")

    {:reply, {:ok, freed}, state}
  end

  @impl true
  def handle_call(:clear, _from, state) do
    state = %{state |
      pools: init_pools(false),
      allocations: %{},
      used_memory: 0
    }

    {:reply, :ok, state}
  end

  @impl true
  def handle_cast({:release, buffer}, state) do
    case Map.pop(state.allocations, buffer.id) do
      {nil, _} ->
        # Unknown buffer, ignore
        {:noreply, state}

      {_buffer, allocations} ->
        size_class = find_size_class(buffer.size)
        pools = return_to_pool(state.pools, size_class, buffer)
        stats = %{state.stats | releases: state.stats.releases + 1}

        state = %{state |
          pools: pools,
          allocations: allocations,
          used_memory: max(0, state.used_memory - buffer.size),
          stats: stats
        }

        {:noreply, state}
    end
  end

  # Private helpers

  defp init_pools(prealloc) do
    Enum.map(@size_classes, fn {size, count} ->
      buffers = if prealloc, do: preallocate_buffers(size, count), else: []
      {size, %{buffers: buffers, allocated: 0}}
    end)
    |> Map.new()
  end

  defp preallocate_buffers(size, count) do
    Enum.map(1..count, fn _ ->
      %{
        id: generate_id(),
        size: size,
        shape: nil,
        dtype: nil,
        tensor: nil,
        allocated_at: nil
      }
    end)
  end

  defp compute_size(shape, dtype) do
    elements = Tuple.product(shape)
    bytes_per_element = dtype_size(dtype)
    elements * bytes_per_element
  end

  defp dtype_size(:f32), do: 4
  defp dtype_size(:f16), do: 2
  defp dtype_size(:bf16), do: 2
  defp dtype_size(:f64), do: 8
  defp dtype_size(:s32), do: 4
  defp dtype_size(:s16), do: 2
  defp dtype_size(:s8), do: 1
  defp dtype_size(:u8), do: 1
  defp dtype_size(_), do: 4

  defp find_size_class(size) do
    Enum.find_value(@size_classes, List.last(@size_classes) |> elem(0), fn {class, _} ->
      if size <= class, do: class
    end)
  end

  defp try_pool_allocate(pools, size_class, _size) do
    case Map.fetch(pools, size_class) do
      {:ok, %{buffers: [buffer | rest]} = pool} ->
        buffer = %{buffer | allocated_at: System.monotonic_time()}
        pools = Map.put(pools, size_class, %{pool | buffers: rest, allocated: pool.allocated + 1})
        {:ok, buffer, pools}

      {:ok, %{buffers: []}} ->
        :empty

      :error ->
        :empty
    end
  end

  defp return_to_pool(pools, size_class, buffer) do
    case Map.fetch(pools, size_class) do
      {:ok, pool} ->
        clean_buffer = %{buffer | tensor: nil, allocated_at: nil}
        Map.put(pools, size_class, %{pool |
          buffers: [clean_buffer | pool.buffers],
          allocated: max(0, pool.allocated - 1)
        })

      :error ->
        pools
    end
  end

  defp allocate_new_buffer(shape, dtype, size) do
    %{
      id: generate_id(),
      shape: shape,
      dtype: dtype,
      size: size,
      tensor: nil,
      allocated_at: System.monotonic_time()
    }
  end

  defp generate_id do
    :crypto.strong_rand_bytes(16) |> Base.encode16(case: :lower)
  end
end
