defmodule Honeycomb.KVCache do
  @moduledoc """
  KV Cache Manager implementing PagedAttention-style block-based memory management.

  This module provides efficient memory management for key-value caches during
  LLM inference by:

  1. **Block-based allocation**: Memory is divided into fixed-size blocks that can
     be allocated and deallocated independently
  2. **Copy-on-write**: Blocks can be shared across requests (e.g., for parallel
     sampling) with copy-on-write semantics
  3. **Prefix sharing**: Common prefixes (like system prompts) can share cache blocks
  4. **Memory pooling**: Blocks are recycled to avoid allocation overhead

  ## Architecture

  The KV cache is organized as:
  - Physical blocks: Actual memory holding KV tensors
  - Logical blocks: Virtual blocks that map to physical blocks
  - Block tables: Per-sequence mapping from logical to physical blocks

  ## Configuration

    * `:block_size` - Number of tokens per block (default: 16)
    * `:num_blocks` - Total number of blocks in the pool
    * `:num_layers` - Number of transformer layers
    * `:num_heads` - Number of attention heads
    * `:head_dim` - Dimension of each attention head
    * `:dtype` - Data type for cache tensors (default: :f16)
  """

  use GenServer
  require Logger

  alias Honeycomb.KVCache.{Block, BlockTable, BlockAllocator}

  @default_block_size 16
  @default_dtype :f16

  defstruct [
    :block_size,
    :num_blocks,
    :num_layers,
    :num_heads,
    :head_dim,
    :dtype,
    :allocator,
    :block_tables,
    :ref_counts,
    :cache_tensors
  ]

  # Client API

  @doc """
  Starts the KV Cache Manager.

  ## Options

    * `:block_size` - Tokens per block (default: 16)
    * `:num_blocks` - Total blocks in pool (required)
    * `:num_layers` - Transformer layers (required)
    * `:num_heads` - Attention heads (required)
    * `:head_dim` - Head dimension (required)
    * `:dtype` - Tensor dtype (default: :f16)
  """
  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Allocates KV cache blocks for a new sequence.

  Returns `{:ok, sequence_id}` or `{:error, :out_of_memory}` if no blocks available.
  """
  def allocate_sequence(num_tokens) do
    GenServer.call(__MODULE__, {:allocate_sequence, num_tokens})
  end

  @doc """
  Extends an existing sequence with additional tokens.

  Returns `:ok` or `{:error, :out_of_memory}`.
  """
  def extend_sequence(sequence_id, num_new_tokens) do
    GenServer.call(__MODULE__, {:extend_sequence, sequence_id, num_new_tokens})
  end

  @doc """
  Frees all blocks associated with a sequence.
  """
  def free_sequence(sequence_id) do
    GenServer.call(__MODULE__, {:free_sequence, sequence_id})
  end

  @doc """
  Forks a sequence for parallel sampling (copy-on-write).

  Returns `{:ok, new_sequence_id}`.
  """
  def fork_sequence(sequence_id) do
    GenServer.call(__MODULE__, {:fork_sequence, sequence_id})
  end

  @doc """
  Gets the block table for a sequence (for attention computation).
  """
  def get_block_table(sequence_id) do
    GenServer.call(__MODULE__, {:get_block_table, sequence_id})
  end

  @doc """
  Gets the KV cache tensors for given block indices.
  """
  def get_cache_tensors(block_indices, layer_idx) do
    GenServer.call(__MODULE__, {:get_cache_tensors, block_indices, layer_idx})
  end

  @doc """
  Updates cache tensors for a specific block and layer.
  """
  def update_cache(block_idx, layer_idx, key_cache, value_cache) do
    GenServer.call(__MODULE__, {:update_cache, block_idx, layer_idx, key_cache, value_cache})
  end

  @doc """
  Returns cache statistics for monitoring.
  """
  def stats do
    GenServer.call(__MODULE__, :stats)
  end

  @doc """
  Returns the number of free blocks available.
  """
  def num_free_blocks do
    GenServer.call(__MODULE__, :num_free_blocks)
  end

  # Server Callbacks

  @impl true
  def init(opts) do
    block_size = Keyword.get(opts, :block_size, @default_block_size)
    num_blocks = Keyword.fetch!(opts, :num_blocks)
    num_layers = Keyword.fetch!(opts, :num_layers)
    num_heads = Keyword.fetch!(opts, :num_heads)
    head_dim = Keyword.fetch!(opts, :head_dim)
    dtype = Keyword.get(opts, :dtype, @default_dtype)

    Logger.info("KVCache: Initializing with #{num_blocks} blocks of size #{block_size}")
    Logger.info("KVCache: Layers=#{num_layers}, Heads=#{num_heads}, HeadDim=#{head_dim}")

    # Initialize the block allocator
    allocator = BlockAllocator.new(num_blocks)

    # Pre-allocate cache tensors for all blocks and layers
    # Shape: [num_blocks, block_size, num_heads, head_dim]
    cache_tensors =
      for layer_idx <- 0..(num_layers - 1), into: %{} do
        key_cache = Nx.broadcast(Nx.tensor(0, type: dtype), {num_blocks, block_size, num_heads, head_dim})
        value_cache = Nx.broadcast(Nx.tensor(0, type: dtype), {num_blocks, block_size, num_heads, head_dim})
        {layer_idx, %{key: key_cache, value: value_cache}}
      end

    state = %__MODULE__{
      block_size: block_size,
      num_blocks: num_blocks,
      num_layers: num_layers,
      num_heads: num_heads,
      head_dim: head_dim,
      dtype: dtype,
      allocator: allocator,
      block_tables: %{},
      ref_counts: %{},
      cache_tensors: cache_tensors
    }

    {:ok, state}
  end

  @impl true
  def handle_call({:allocate_sequence, num_tokens}, _from, state) do
    num_blocks_needed = blocks_needed(num_tokens, state.block_size)

    case BlockAllocator.allocate(state.allocator, num_blocks_needed) do
      {:ok, block_indices, allocator} ->
        sequence_id = generate_sequence_id()
        block_table = BlockTable.new(block_indices, state.block_size)

        # Initialize ref counts for new blocks
        ref_counts =
          Enum.reduce(block_indices, state.ref_counts, fn idx, acc ->
            Map.update(acc, idx, 1, &(&1 + 1))
          end)

        state = %{state |
          allocator: allocator,
          block_tables: Map.put(state.block_tables, sequence_id, block_table),
          ref_counts: ref_counts
        }

        {:reply, {:ok, sequence_id}, state}

      {:error, :out_of_memory} = error ->
        {:reply, error, state}
    end
  end

  @impl true
  def handle_call({:extend_sequence, sequence_id, num_new_tokens}, _from, state) do
    case Map.fetch(state.block_tables, sequence_id) do
      {:ok, block_table} ->
        current_tokens = BlockTable.num_tokens(block_table)
        new_total = current_tokens + num_new_tokens
        additional_blocks = blocks_needed(new_total, state.block_size) - length(block_table.block_indices)

        if additional_blocks > 0 do
          case BlockAllocator.allocate(state.allocator, additional_blocks) do
            {:ok, new_indices, allocator} ->
              block_table = BlockTable.extend(block_table, new_indices, num_new_tokens)

              ref_counts =
                Enum.reduce(new_indices, state.ref_counts, fn idx, acc ->
                  Map.update(acc, idx, 1, &(&1 + 1))
                end)

              state = %{state |
                allocator: allocator,
                block_tables: Map.put(state.block_tables, sequence_id, block_table),
                ref_counts: ref_counts
              }

              {:reply, :ok, state}

            {:error, :out_of_memory} = error ->
              {:reply, error, state}
          end
        else
          block_table = BlockTable.add_tokens(block_table, num_new_tokens)
          state = %{state | block_tables: Map.put(state.block_tables, sequence_id, block_table)}
          {:reply, :ok, state}
        end

      :error ->
        {:reply, {:error, :unknown_sequence}, state}
    end
  end

  @impl true
  def handle_call({:free_sequence, sequence_id}, _from, state) do
    case Map.fetch(state.block_tables, sequence_id) do
      {:ok, block_table} ->
        {allocator, ref_counts} =
          Enum.reduce(block_table.block_indices, {state.allocator, state.ref_counts}, fn idx, {alloc, refs} ->
            new_count = Map.get(refs, idx, 1) - 1

            if new_count <= 0 do
              {BlockAllocator.free(alloc, idx), Map.delete(refs, idx)}
            else
              {alloc, Map.put(refs, idx, new_count)}
            end
          end)

        state = %{state |
          allocator: allocator,
          block_tables: Map.delete(state.block_tables, sequence_id),
          ref_counts: ref_counts
        }

        {:reply, :ok, state}

      :error ->
        {:reply, {:error, :unknown_sequence}, state}
    end
  end

  @impl true
  def handle_call({:fork_sequence, sequence_id}, _from, state) do
    case Map.fetch(state.block_tables, sequence_id) do
      {:ok, block_table} ->
        new_sequence_id = generate_sequence_id()

        # Copy-on-write: just increment ref counts
        ref_counts =
          Enum.reduce(block_table.block_indices, state.ref_counts, fn idx, acc ->
            Map.update(acc, idx, 1, &(&1 + 1))
          end)

        state = %{state |
          block_tables: Map.put(state.block_tables, new_sequence_id, block_table),
          ref_counts: ref_counts
        }

        {:reply, {:ok, new_sequence_id}, state}

      :error ->
        {:reply, {:error, :unknown_sequence}, state}
    end
  end

  @impl true
  def handle_call({:get_block_table, sequence_id}, _from, state) do
    case Map.fetch(state.block_tables, sequence_id) do
      {:ok, block_table} ->
        {:reply, {:ok, block_table}, state}

      :error ->
        {:reply, {:error, :unknown_sequence}, state}
    end
  end

  @impl true
  def handle_call({:get_cache_tensors, block_indices, layer_idx}, _from, state) do
    case Map.fetch(state.cache_tensors, layer_idx) do
      {:ok, %{key: key_cache, value: value_cache}} ->
        # Gather cache values for the requested blocks
        indices_tensor = Nx.tensor(block_indices)
        keys = Nx.take(key_cache, indices_tensor, axis: 0)
        values = Nx.take(value_cache, indices_tensor, axis: 0)
        {:reply, {:ok, keys, values}, state}

      :error ->
        {:reply, {:error, :invalid_layer}, state}
    end
  end

  @impl true
  def handle_call({:update_cache, block_idx, layer_idx, new_keys, new_values}, _from, state) do
    case Map.fetch(state.cache_tensors, layer_idx) do
      {:ok, %{key: key_cache, value: value_cache}} ->
        # Check if copy-on-write is needed
        ref_count = Map.get(state.ref_counts, block_idx, 0)

        {block_idx, state} =
          if ref_count > 1 do
            # Need to copy block before writing
            copy_block_on_write(block_idx, layer_idx, state)
          else
            {block_idx, state}
          end

        # Update the cache tensors
        key_cache = Nx.put_slice(key_cache, [block_idx, 0, 0, 0], new_keys)
        value_cache = Nx.put_slice(value_cache, [block_idx, 0, 0, 0], new_values)

        cache_tensors = Map.put(state.cache_tensors, layer_idx, %{key: key_cache, value: value_cache})
        state = %{state | cache_tensors: cache_tensors}

        {:reply, :ok, state}

      :error ->
        {:reply, {:error, :invalid_layer}, state}
    end
  end

  @impl true
  def handle_call(:stats, _from, state) do
    stats = %{
      total_blocks: state.num_blocks,
      free_blocks: BlockAllocator.num_free(state.allocator),
      used_blocks: state.num_blocks - BlockAllocator.num_free(state.allocator),
      active_sequences: map_size(state.block_tables),
      block_size: state.block_size,
      memory_usage_mb: calculate_memory_usage(state)
    }

    {:reply, stats, state}
  end

  @impl true
  def handle_call(:num_free_blocks, _from, state) do
    {:reply, BlockAllocator.num_free(state.allocator), state}
  end

  # Private functions

  defp blocks_needed(num_tokens, block_size) do
    div(num_tokens + block_size - 1, block_size)
  end

  defp generate_sequence_id do
    :crypto.strong_rand_bytes(16) |> Base.encode16(case: :lower)
  end

  defp copy_block_on_write(block_idx, _layer_idx, state) do
    case BlockAllocator.allocate(state.allocator, 1) do
      {:ok, [new_block_idx], allocator} ->
        # Copy data from old block to new block for all layers
        cache_tensors =
          Enum.reduce(0..(state.num_layers - 1), state.cache_tensors, fn layer, acc ->
            %{key: key_cache, value: value_cache} = Map.fetch!(acc, layer)

            # Extract old block data
            old_key = Nx.slice(key_cache, [block_idx, 0, 0, 0], [1, state.block_size, state.num_heads, state.head_dim])
            old_value = Nx.slice(value_cache, [block_idx, 0, 0, 0], [1, state.block_size, state.num_heads, state.head_dim])

            # Write to new block
            key_cache = Nx.put_slice(key_cache, [new_block_idx, 0, 0, 0], old_key)
            value_cache = Nx.put_slice(value_cache, [new_block_idx, 0, 0, 0], old_value)

            Map.put(acc, layer, %{key: key_cache, value: value_cache})
          end)

        # Update ref count for old block
        ref_counts = Map.update!(state.ref_counts, block_idx, &(&1 - 1))
        ref_counts = Map.put(ref_counts, new_block_idx, 1)

        state = %{state |
          allocator: allocator,
          cache_tensors: cache_tensors,
          ref_counts: ref_counts
        }

        {new_block_idx, state}

      {:error, :out_of_memory} ->
        # Fall back to in-place update if out of memory
        Logger.warning("KVCache: Out of memory for copy-on-write, falling back to shared write")
        {block_idx, state}
    end
  end

  defp calculate_memory_usage(state) do
    bytes_per_element =
      case state.dtype do
        :f16 -> 2
        :bf16 -> 2
        :f32 -> 4
        :f64 -> 8
        _ -> 2
      end

    elements_per_block = state.block_size * state.num_heads * state.head_dim
    # 2 for key and value caches
    total_elements = state.num_blocks * elements_per_block * 2 * state.num_layers
    total_elements * bytes_per_element / (1024 * 1024)
  end
end
