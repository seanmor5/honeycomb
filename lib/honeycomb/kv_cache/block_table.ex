defmodule Honeycomb.KVCache.BlockTable do
  @moduledoc """
  Block table for mapping logical sequence positions to physical cache blocks.

  Each sequence maintains a block table that maps token positions to
  physical blocks in the KV cache. This enables:
  - Efficient lookup of cache data during attention
  - Copy-on-write for parallel sampling
  - Prefix sharing across sequences
  """

  defstruct [:block_indices, :block_size, :num_tokens, :slot_mapping]

  @doc """
  Creates a new block table for a sequence.
  """
  def new(block_indices, block_size) when is_list(block_indices) and block_size > 0 do
    %__MODULE__{
      block_indices: block_indices,
      block_size: block_size,
      num_tokens: 0,
      slot_mapping: []
    }
  end

  @doc """
  Returns the number of tokens currently stored.
  """
  def num_tokens(%__MODULE__{num_tokens: n}), do: n

  @doc """
  Returns the total capacity (max tokens that can be stored).
  """
  def capacity(%__MODULE__{block_indices: indices, block_size: bs}) do
    length(indices) * bs
  end

  @doc """
  Returns the remaining capacity.
  """
  def remaining_capacity(%__MODULE__{} = table) do
    capacity(table) - table.num_tokens
  end

  @doc """
  Extends the block table with additional blocks.
  """
  def extend(%__MODULE__{} = table, new_indices, additional_tokens) when is_list(new_indices) do
    %{table |
      block_indices: table.block_indices ++ new_indices,
      num_tokens: table.num_tokens + additional_tokens
    }
  end

  @doc """
  Adds tokens without allocating new blocks (must have capacity).
  """
  def add_tokens(%__MODULE__{} = table, num_tokens) do
    new_total = table.num_tokens + num_tokens

    if new_total > capacity(table) do
      raise "Cannot add #{num_tokens} tokens: would exceed capacity"
    end

    %{table | num_tokens: new_total}
  end

  @doc """
  Gets the physical block index and slot offset for a given token position.
  """
  def get_slot(%__MODULE__{} = table, token_pos) when token_pos >= 0 do
    if token_pos >= table.num_tokens do
      {:error, :out_of_bounds}
    else
      block_idx = div(token_pos, table.block_size)
      slot_offset = rem(token_pos, table.block_size)
      physical_block = Enum.at(table.block_indices, block_idx)
      {:ok, physical_block, slot_offset}
    end
  end

  @doc """
  Gets the slot mapping for a range of token positions.

  Returns a list of `{physical_block, slot_offset}` tuples.
  """
  def get_slot_range(%__MODULE__{} = table, start_pos, end_pos) do
    for pos <- start_pos..(end_pos - 1) do
      {:ok, block, offset} = get_slot(table, pos)
      {block, offset}
    end
  end

  @doc """
  Returns the block indices as a tensor for attention computation.
  """
  def to_tensor(%__MODULE__{block_indices: indices}) do
    Nx.tensor(indices, type: :s32)
  end

  @doc """
  Returns physical slot indices for all tokens (for scatter/gather operations).
  """
  def all_slots(%__MODULE__{} = table) do
    for pos <- 0..(table.num_tokens - 1) do
      block_idx = div(pos, table.block_size)
      slot_offset = rem(pos, table.block_size)
      physical_block = Enum.at(table.block_indices, block_idx)
      physical_block * table.block_size + slot_offset
    end
  end

  @doc """
  Computes the slot index for the next token to be added.
  """
  def next_slot(%__MODULE__{} = table) do
    if table.num_tokens >= capacity(table) do
      {:error, :no_capacity}
    else
      pos = table.num_tokens
      block_idx = div(pos, table.block_size)
      slot_offset = rem(pos, table.block_size)
      physical_block = Enum.at(table.block_indices, block_idx)
      {:ok, physical_block, slot_offset}
    end
  end

  @doc """
  Creates a copy of the block table (for forking sequences).
  """
  def copy(%__MODULE__{} = table) do
    %{table | slot_mapping: []}
  end

  @doc """
  Returns the number of blocks used.
  """
  def num_blocks(%__MODULE__{block_indices: indices}), do: length(indices)
end
