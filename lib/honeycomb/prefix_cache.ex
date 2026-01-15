defmodule Honeycomb.PrefixCache do
  @moduledoc """
  Prefix Caching using a radix tree for efficient prefix matching.

  This module enables reuse of computed KV cache for common prefixes
  (like system prompts) across different requests. Key features:

  1. **Radix tree structure**: Efficient prefix matching and storage
  2. **Hash-based lookup**: Fast O(1) lookup for exact prefixes
  3. **LRU eviction**: Least-recently-used eviction when cache is full
  4. **Block-aligned**: Prefixes are stored at block boundaries

  ## How It Works

  When a new request arrives:
  1. Compute hash of the prefix tokens
  2. Look up in radix tree to find longest matching cached prefix
  3. If found, reuse the cached KV blocks
  4. Only compute KV for the non-cached portion

  This is particularly effective for:
  - System prompts (often identical across requests)
  - Multi-turn conversations (reuse previous turns)
  - Few-shot examples in prompts
  """

  use GenServer
  require Logger

  alias Honeycomb.PrefixCache.RadixTree

  defstruct [
    :tree,
    :hash_to_node,
    :block_size,
    :max_cached_blocks,
    :lru_list,
    :stats
  ]

  # Client API

  @doc """
  Starts the prefix cache.

  ## Options

    * `:block_size` - KV cache block size (default: 16)
    * `:max_cached_blocks` - Maximum blocks to cache (default: 1000)
  """
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Looks up a prefix in the cache.

  Returns `{:ok, block_indices, matched_length}` if found,
  or `{:miss, 0}` if not cached.
  """
  def lookup(token_ids) when is_list(token_ids) do
    GenServer.call(__MODULE__, {:lookup, token_ids})
  end

  @doc """
  Adds a prefix to the cache.

  Associates the token sequence with the given KV cache block indices.
  """
  def insert(token_ids, block_indices) when is_list(token_ids) and is_list(block_indices) do
    GenServer.call(__MODULE__, {:insert, token_ids, block_indices})
  end

  @doc """
  Removes a prefix from the cache.
  """
  def evict(token_ids) when is_list(token_ids) do
    GenServer.call(__MODULE__, {:evict, token_ids})
  end

  @doc """
  Returns cache statistics.
  """
  def stats do
    GenServer.call(__MODULE__, :stats)
  end

  @doc """
  Clears the entire cache.
  """
  def clear do
    GenServer.call(__MODULE__, :clear)
  end

  @doc """
  Computes the hash of a token sequence for cache lookup.
  """
  def hash_tokens(token_ids) when is_list(token_ids) do
    :erlang.phash2(token_ids)
  end

  # Server Callbacks

  @impl true
  def init(opts) do
    block_size = Keyword.get(opts, :block_size, 16)
    max_cached_blocks = Keyword.get(opts, :max_cached_blocks, 1000)

    state = %__MODULE__{
      tree: RadixTree.new(),
      hash_to_node: %{},
      block_size: block_size,
      max_cached_blocks: max_cached_blocks,
      lru_list: [],
      stats: %{hits: 0, misses: 0, insertions: 0, evictions: 0}
    }

    Logger.info("PrefixCache: Started with max_blocks=#{max_cached_blocks}, block_size=#{block_size}")

    {:ok, state}
  end

  @impl true
  def handle_call({:lookup, token_ids}, _from, state) do
    hash = hash_tokens(token_ids)

    case Map.fetch(state.hash_to_node, hash) do
      {:ok, %{tokens: cached_tokens, blocks: blocks}} when cached_tokens == token_ids ->
        # Exact match - move to front of LRU
        lru_list = [hash | List.delete(state.lru_list, hash)]
        stats = %{state.stats | hits: state.stats.hits + 1}
        state = %{state | lru_list: lru_list, stats: stats}
        {:reply, {:ok, blocks, length(token_ids)}, state}

      _ ->
        # Try prefix match in radix tree
        case RadixTree.find_longest_prefix(state.tree, token_ids) do
          {:ok, matched_tokens, blocks} ->
            # Partial match
            hash = hash_tokens(matched_tokens)
            lru_list = [hash | List.delete(state.lru_list, hash)]
            stats = %{state.stats | hits: state.stats.hits + 1}
            state = %{state | lru_list: lru_list, stats: stats}
            {:reply, {:ok, blocks, length(matched_tokens)}, state}

          :not_found ->
            stats = %{state.stats | misses: state.stats.misses + 1}
            state = %{state | stats: stats}
            {:reply, {:miss, 0}, state}
        end
    end
  end

  @impl true
  def handle_call({:insert, token_ids, block_indices}, _from, state) do
    # Evict if necessary
    state = maybe_evict(state, length(block_indices))

    hash = hash_tokens(token_ids)
    node = %{tokens: token_ids, blocks: block_indices}

    tree = RadixTree.insert(state.tree, token_ids, block_indices)
    hash_to_node = Map.put(state.hash_to_node, hash, node)
    lru_list = [hash | state.lru_list]
    stats = %{state.stats | insertions: state.stats.insertions + 1}

    state = %{state |
      tree: tree,
      hash_to_node: hash_to_node,
      lru_list: lru_list,
      stats: stats
    }

    {:reply, :ok, state}
  end

  @impl true
  def handle_call({:evict, token_ids}, _from, state) do
    hash = hash_tokens(token_ids)

    tree = RadixTree.delete(state.tree, token_ids)
    hash_to_node = Map.delete(state.hash_to_node, hash)
    lru_list = List.delete(state.lru_list, hash)
    stats = %{state.stats | evictions: state.stats.evictions + 1}

    state = %{state |
      tree: tree,
      hash_to_node: hash_to_node,
      lru_list: lru_list,
      stats: stats
    }

    {:reply, :ok, state}
  end

  @impl true
  def handle_call(:stats, _from, state) do
    total_lookups = state.stats.hits + state.stats.misses
    hit_rate = if total_lookups > 0, do: state.stats.hits / total_lookups, else: 0.0

    stats = Map.merge(state.stats, %{
      cached_prefixes: map_size(state.hash_to_node),
      hit_rate: hit_rate
    })

    {:reply, stats, state}
  end

  @impl true
  def handle_call(:clear, _from, state) do
    state = %{state |
      tree: RadixTree.new(),
      hash_to_node: %{},
      lru_list: []
    }

    {:reply, :ok, state}
  end

  # Private helpers

  defp maybe_evict(state, needed_blocks) do
    current_blocks = count_cached_blocks(state)

    if current_blocks + needed_blocks > state.max_cached_blocks do
      evict_lru(state, needed_blocks)
    else
      state
    end
  end

  defp count_cached_blocks(state) do
    state.hash_to_node
    |> Map.values()
    |> Enum.reduce(0, fn %{blocks: blocks}, acc -> acc + length(blocks) end)
  end

  defp evict_lru(state, needed_blocks) do
    {state, _freed} = do_evict_lru(state, needed_blocks, 0)
    state
  end

  defp do_evict_lru(state, needed, freed) when freed >= needed, do: {state, freed}

  defp do_evict_lru(%{lru_list: []} = state, _needed, freed), do: {state, freed}

  defp do_evict_lru(state, needed, freed) do
    # Evict least recently used (end of list)
    [oldest | rest] = Enum.reverse(state.lru_list)

    case Map.fetch(state.hash_to_node, oldest) do
      {:ok, %{tokens: tokens, blocks: blocks}} ->
        tree = RadixTree.delete(state.tree, tokens)
        hash_to_node = Map.delete(state.hash_to_node, oldest)
        stats = %{state.stats | evictions: state.stats.evictions + 1}

        state = %{state |
          tree: tree,
          hash_to_node: hash_to_node,
          lru_list: Enum.reverse(rest),
          stats: stats
        }

        do_evict_lru(state, needed, freed + length(blocks))

      :error ->
        state = %{state | lru_list: Enum.reverse(rest)}
        do_evict_lru(state, needed, freed)
    end
  end
end
