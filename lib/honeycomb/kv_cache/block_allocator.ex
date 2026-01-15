defmodule Honeycomb.KVCache.BlockAllocator do
  @moduledoc """
  Block allocator for KV cache memory management.

  Manages a pool of fixed-size blocks using a free list. Supports:
  - O(1) allocation when blocks are available
  - O(1) deallocation
  - Batch allocation for multiple blocks
  """

  defstruct [:num_blocks, :free_list, :num_free]

  @doc """
  Creates a new block allocator with the given number of blocks.
  """
  def new(num_blocks) when num_blocks > 0 do
    # Initialize free list with all block indices
    free_list = :queue.from_list(Enum.to_list(0..(num_blocks - 1)))

    %__MODULE__{
      num_blocks: num_blocks,
      free_list: free_list,
      num_free: num_blocks
    }
  end

  @doc """
  Allocates the specified number of blocks.

  Returns `{:ok, block_indices, updated_allocator}` or `{:error, :out_of_memory}`.
  """
  def allocate(%__MODULE__{} = allocator, num_blocks) when num_blocks > 0 do
    if allocator.num_free >= num_blocks do
      {indices, free_list} = pop_n(allocator.free_list, num_blocks, [])

      allocator = %{allocator |
        free_list: free_list,
        num_free: allocator.num_free - num_blocks
      }

      {:ok, Enum.reverse(indices), allocator}
    else
      {:error, :out_of_memory}
    end
  end

  def allocate(%__MODULE__{} = allocator, 0), do: {:ok, [], allocator}

  @doc """
  Frees a single block, returning it to the pool.
  """
  def free(%__MODULE__{} = allocator, block_idx) when is_integer(block_idx) do
    %{allocator |
      free_list: :queue.in(block_idx, allocator.free_list),
      num_free: allocator.num_free + 1
    }
  end

  @doc """
  Frees multiple blocks at once.
  """
  def free_many(%__MODULE__{} = allocator, block_indices) when is_list(block_indices) do
    Enum.reduce(block_indices, allocator, fn idx, acc -> free(acc, idx) end)
  end

  @doc """
  Returns the number of free blocks.
  """
  def num_free(%__MODULE__{num_free: num_free}), do: num_free

  @doc """
  Returns the total number of blocks.
  """
  def num_total(%__MODULE__{num_blocks: num_blocks}), do: num_blocks

  @doc """
  Returns the utilization ratio (0.0 to 1.0).
  """
  def utilization(%__MODULE__{} = allocator) do
    (allocator.num_blocks - allocator.num_free) / allocator.num_blocks
  end

  @doc """
  Checks if allocation of n blocks would succeed.
  """
  def can_allocate?(%__MODULE__{num_free: num_free}, num_blocks) do
    num_free >= num_blocks
  end

  # Private helpers

  defp pop_n(queue, 0, acc), do: {acc, queue}

  defp pop_n(queue, n, acc) do
    case :queue.out(queue) do
      {{:value, item}, rest} ->
        pop_n(rest, n - 1, [item | acc])

      {:empty, _} ->
        {acc, queue}
    end
  end
end
