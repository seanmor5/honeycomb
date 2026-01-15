defmodule Honeycomb.PrefixCache.RadixTree do
  @moduledoc """
  Radix tree (Patricia trie) implementation for prefix matching.

  Optimized for token sequence lookups where we want to find
  the longest matching prefix efficiently.
  """

  defstruct [:children, :value, :is_terminal]

  @doc """
  Creates a new empty radix tree.
  """
  def new do
    %__MODULE__{
      children: %{},
      value: nil,
      is_terminal: false
    }
  end

  @doc """
  Inserts a key-value pair into the tree.
  """
  def insert(%__MODULE__{} = tree, [], value) do
    %{tree | value: value, is_terminal: true}
  end

  def insert(%__MODULE__{} = tree, [head | tail], value) do
    child = Map.get(tree.children, head, new())
    updated_child = insert(child, tail, value)
    %{tree | children: Map.put(tree.children, head, updated_child)}
  end

  @doc """
  Looks up an exact key in the tree.
  """
  def lookup(%__MODULE__{} = tree, []) do
    if tree.is_terminal do
      {:ok, tree.value}
    else
      :not_found
    end
  end

  def lookup(%__MODULE__{} = tree, [head | tail]) do
    case Map.fetch(tree.children, head) do
      {:ok, child} -> lookup(child, tail)
      :error -> :not_found
    end
  end

  @doc """
  Finds the longest matching prefix and returns its value.

  Returns `{:ok, matched_tokens, value}` or `:not_found`.
  """
  def find_longest_prefix(%__MODULE__{} = tree, tokens) do
    find_longest_prefix(tree, tokens, [], nil)
  end

  defp find_longest_prefix(%__MODULE__{} = tree, [], matched, last_match) do
    if tree.is_terminal do
      {:ok, Enum.reverse(matched), tree.value}
    else
      case last_match do
        nil -> :not_found
        {matched_tokens, value} -> {:ok, matched_tokens, value}
      end
    end
  end

  defp find_longest_prefix(%__MODULE__{} = tree, [head | tail], matched, last_match) do
    # Update last_match if current node is terminal
    last_match =
      if tree.is_terminal do
        {Enum.reverse(matched), tree.value}
      else
        last_match
      end

    case Map.fetch(tree.children, head) do
      {:ok, child} ->
        find_longest_prefix(child, tail, [head | matched], last_match)

      :error ->
        # No more matches, return the last terminal we found
        if tree.is_terminal do
          {:ok, Enum.reverse(matched), tree.value}
        else
          case last_match do
            nil -> :not_found
            {matched_tokens, value} -> {:ok, matched_tokens, value}
          end
        end
    end
  end

  @doc """
  Deletes a key from the tree.
  """
  def delete(%__MODULE__{} = tree, []) do
    %{tree | value: nil, is_terminal: false}
  end

  def delete(%__MODULE__{} = tree, [head | tail]) do
    case Map.fetch(tree.children, head) do
      {:ok, child} ->
        updated_child = delete(child, tail)

        # Remove child if it's now empty
        children =
          if empty?(updated_child) do
            Map.delete(tree.children, head)
          else
            Map.put(tree.children, head, updated_child)
          end

        %{tree | children: children}

      :error ->
        tree
    end
  end

  @doc """
  Checks if a node is empty (no children and not terminal).
  """
  def empty?(%__MODULE__{children: children, is_terminal: false}) when map_size(children) == 0 do
    true
  end

  def empty?(%__MODULE__{}), do: false

  @doc """
  Returns all keys in the tree.
  """
  def keys(%__MODULE__{} = tree) do
    collect_keys(tree, [], [])
  end

  defp collect_keys(%__MODULE__{} = tree, current_key, acc) do
    acc =
      if tree.is_terminal do
        [Enum.reverse(current_key) | acc]
      else
        acc
      end

    Enum.reduce(tree.children, acc, fn {token, child}, acc ->
      collect_keys(child, [token | current_key], acc)
    end)
  end

  @doc """
  Returns the number of entries in the tree.
  """
  def size(%__MODULE__{} = tree) do
    count_terminals(tree)
  end

  defp count_terminals(%__MODULE__{} = tree) do
    base = if tree.is_terminal, do: 1, else: 0

    Enum.reduce(tree.children, base, fn {_token, child}, acc ->
      acc + count_terminals(child)
    end)
  end

  @doc """
  Checks if a prefix exists in the tree.
  """
  def has_prefix?(%__MODULE__{} = tree, tokens) do
    case find_longest_prefix(tree, tokens) do
      {:ok, matched, _value} -> length(matched) > 0
      :not_found -> false
    end
  end
end
