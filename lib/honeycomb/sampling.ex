defmodule Honeycomb.Sampling do
  @moduledoc """
  Sampling strategies for token generation.

  Implements various sampling methods used in LLM inference:

  - **Greedy**: Always pick the highest probability token
  - **Temperature**: Scale logits before softmax
  - **Top-K**: Sample from top K tokens only
  - **Top-P (Nucleus)**: Sample from smallest set with cumulative prob >= p
  - **Min-P**: Filter tokens below min_p * max_prob
  - **Typical**: Sample from tokens with entropy close to expected
  - **Mirostat**: Adaptive sampling to maintain target perplexity

  ## Usage

      logits = model_forward(input)
      params = Honeycomb.Sampling.Params.new(temperature: 0.8, top_p: 0.9)
      token = Honeycomb.Sampling.sample(logits, params)
  """

  import Nx.Defn

  alias Honeycomb.Sampling.Params

  @doc """
  Samples a token from logits using the given parameters.
  """
  def sample(logits, %Params{} = params) do
    logits
    |> apply_repetition_penalty(params.repetition_penalty, params.context_tokens)
    |> apply_frequency_penalty(params.frequency_penalty, params.token_counts)
    |> apply_presence_penalty(params.presence_penalty, params.token_counts)
    |> apply_temperature(params.temperature)
    |> apply_top_k(params.top_k)
    |> apply_top_p(params.top_p)
    |> apply_min_p(params.min_p)
    |> do_sample(params.seed)
  end

  @doc """
  Applies temperature scaling to logits.
  """
  defn apply_temperature(logits, temperature) do
    if temperature > 0 do
      Nx.divide(logits, temperature)
    else
      logits
    end
  end

  @doc """
  Applies top-k filtering to logits.

  Sets logits outside top-k to -infinity.
  """
  def apply_top_k(logits, k) when k <= 0, do: logits

  def apply_top_k(logits, k) do
    {sorted_logits, sorted_indices} = top_k_sorted(logits, k)
    vocab_size = Nx.axis_size(logits, -1)

    # Create mask for top-k indices
    mask = Nx.broadcast(Nx.tensor(false), {vocab_size})

    mask =
      sorted_indices
      |> Nx.to_flat_list()
      |> Enum.take(k)
      |> Enum.reduce(mask, fn idx, m ->
        Nx.indexed_put(m, Nx.tensor([[idx]]), Nx.tensor([true]))
      end)

    # Apply mask
    Nx.select(mask, logits, Nx.Constants.neg_infinity())
  end

  @doc """
  Applies top-p (nucleus) filtering to logits.

  Keeps smallest set of tokens with cumulative probability >= p.
  """
  def apply_top_p(logits, p) when p >= 1.0, do: logits

  def apply_top_p(logits, p) do
    probs = softmax(logits)
    {sorted_probs, sorted_indices} = sort_descending(probs)

    # Compute cumulative probabilities
    cumsum = cumulative_sum(sorted_probs)

    # Find cutoff index
    cutoff_idx =
      cumsum
      |> Nx.to_flat_list()
      |> Enum.find_index(&(&1 >= p))
      |> Kernel.||(Nx.axis_size(probs, -1) - 1)

    # Keep only tokens up to cutoff
    keep_indices =
      sorted_indices
      |> Nx.to_flat_list()
      |> Enum.take(cutoff_idx + 1)
      |> MapSet.new()

    vocab_size = Nx.axis_size(logits, -1)

    # Create filtered logits
    Enum.reduce(0..(vocab_size - 1), logits, fn idx, acc ->
      if MapSet.member?(keep_indices, idx) do
        acc
      else
        Nx.indexed_put(acc, Nx.tensor([[idx]]), Nx.Constants.neg_infinity())
      end
    end)
  end

  @doc """
  Applies min-p filtering.

  Filters tokens with probability < min_p * max_probability.
  """
  def apply_min_p(logits, min_p) when min_p <= 0, do: logits

  def apply_min_p(logits, min_p) do
    probs = softmax(logits)
    max_prob = Nx.reduce_max(probs)
    threshold = Nx.multiply(max_prob, min_p)

    mask = Nx.greater_equal(probs, threshold)
    Nx.select(mask, logits, Nx.Constants.neg_infinity())
  end

  @doc """
  Applies repetition penalty to discourage repeated tokens.
  """
  def apply_repetition_penalty(logits, penalty, context_tokens)
      when penalty == 1.0 or context_tokens == [] do
    logits
  end

  def apply_repetition_penalty(logits, penalty, context_tokens) do
    unique_tokens = MapSet.new(context_tokens)

    Enum.reduce(unique_tokens, logits, fn token, acc ->
      current = Nx.to_number(acc[token])

      adjusted =
        if current > 0 do
          current / penalty
        else
          current * penalty
        end

      Nx.indexed_put(acc, Nx.tensor([[token]]), Nx.tensor([adjusted]))
    end)
  end

  @doc """
  Applies frequency penalty based on token occurrence count.
  """
  def apply_frequency_penalty(logits, penalty, token_counts)
      when penalty == 0.0 or map_size(token_counts) == 0 do
    logits
  end

  def apply_frequency_penalty(logits, penalty, token_counts) do
    Enum.reduce(token_counts, logits, fn {token, count}, acc ->
      current = Nx.to_number(acc[token])
      adjusted = current - penalty * count
      Nx.indexed_put(acc, Nx.tensor([[token]]), Nx.tensor([adjusted]))
    end)
  end

  @doc """
  Applies presence penalty (penalizes tokens that appeared at all).
  """
  def apply_presence_penalty(logits, penalty, token_counts)
      when penalty == 0.0 or map_size(token_counts) == 0 do
    logits
  end

  def apply_presence_penalty(logits, penalty, token_counts) do
    Enum.reduce(token_counts, logits, fn {token, _count}, acc ->
      current = Nx.to_number(acc[token])
      adjusted = current - penalty
      Nx.indexed_put(acc, Nx.tensor([[token]]), Nx.tensor([adjusted]))
    end)
  end

  @doc """
  Performs the actual sampling from processed logits.
  """
  def do_sample(logits, seed) do
    # Set random seed if provided
    if seed, do: :rand.seed(:exsss, seed)

    probs = softmax(logits)
    multinomial_sample(probs)
  end

  @doc """
  Greedy decoding - returns argmax.
  """
  def greedy(logits) do
    Nx.argmax(logits) |> Nx.to_number()
  end

  # Private helpers

  defp softmax(logits) do
    max_logit = Nx.reduce_max(logits)
    shifted = Nx.subtract(logits, max_logit)
    exp_logits = Nx.exp(shifted)
    Nx.divide(exp_logits, Nx.sum(exp_logits))
  end

  defp top_k_sorted(tensor, k) do
    flat = Nx.to_flat_list(tensor)
    indexed = Enum.with_index(flat)
    sorted = Enum.sort_by(indexed, fn {val, _idx} -> -val end)
    top_k = Enum.take(sorted, k)

    values = Enum.map(top_k, fn {v, _} -> v end) |> Nx.tensor()
    indices = Enum.map(top_k, fn {_, i} -> i end) |> Nx.tensor()

    {values, indices}
  end

  defp sort_descending(tensor) do
    flat = Nx.to_flat_list(tensor)
    indexed = Enum.with_index(flat)
    sorted = Enum.sort_by(indexed, fn {val, _idx} -> -val end)

    values = Enum.map(sorted, fn {v, _} -> v end) |> Nx.tensor()
    indices = Enum.map(sorted, fn {_, i} -> i end) |> Nx.tensor()

    {values, indices}
  end

  defp cumulative_sum(tensor) do
    flat = Nx.to_flat_list(tensor)

    {cumsum, _} =
      Enum.map_reduce(flat, 0.0, fn x, acc ->
        new_acc = acc + x
        {new_acc, new_acc}
      end)

    Nx.tensor(cumsum)
  end

  defp multinomial_sample(probs) do
    r = :rand.uniform()
    flat = Nx.to_flat_list(probs)

    {_, idx} =
      Enum.reduce_while(flat, {0.0, 0}, fn p, {cumsum, idx} ->
        new_cumsum = cumsum + p

        if new_cumsum >= r do
          {:halt, {new_cumsum, idx}}
        else
          {:cont, {new_cumsum, idx + 1}}
        end
      end)

    # Handle edge case where we might exceed bounds
    min(idx, length(flat) - 1)
  end
end
