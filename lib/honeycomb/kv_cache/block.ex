defmodule Honeycomb.KVCache.Block do
  @moduledoc """
  Represents a single block in the KV cache.

  A block holds a fixed number of token KV pairs and tracks its usage state.
  """

  defstruct [:id, :size, :num_filled, :ref_count]

  @doc """
  Creates a new empty block.
  """
  def new(id, size) when is_integer(id) and size > 0 do
    %__MODULE__{
      id: id,
      size: size,
      num_filled: 0,
      ref_count: 0
    }
  end

  @doc """
  Checks if the block is full.
  """
  def full?(%__MODULE__{num_filled: filled, size: size}), do: filled >= size

  @doc """
  Checks if the block is empty.
  """
  def empty?(%__MODULE__{num_filled: 0}), do: true
  def empty?(%__MODULE__{}), do: false

  @doc """
  Returns remaining slots in the block.
  """
  def remaining(%__MODULE__{num_filled: filled, size: size}), do: size - filled

  @doc """
  Increments the reference count (for copy-on-write).
  """
  def inc_ref(%__MODULE__{} = block) do
    %{block | ref_count: block.ref_count + 1}
  end

  @doc """
  Decrements the reference count.
  """
  def dec_ref(%__MODULE__{ref_count: count} = block) when count > 0 do
    %{block | ref_count: count - 1}
  end

  @doc """
  Checks if the block is shared (ref_count > 1).
  """
  def shared?(%__MODULE__{ref_count: count}), do: count > 1

  @doc """
  Marks n slots as filled.
  """
  def fill(%__MODULE__{} = block, n) when n > 0 do
    new_filled = min(block.num_filled + n, block.size)
    %{block | num_filled: new_filled}
  end

  @doc """
  Resets the block to empty state.
  """
  def reset(%__MODULE__{} = block) do
    %{block | num_filled: 0, ref_count: 0}
  end
end
