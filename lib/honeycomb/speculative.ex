defmodule Honeycomb.Speculative do
  @moduledoc """
  Speculative Decoding implementation for faster inference.

  Speculative decoding uses a smaller, faster "draft" model to propose
  multiple tokens at once, which are then verified by the target model
  in a single forward pass. This can provide significant speedups when:

  1. The draft model is much faster than the target model
  2. The draft model has good accuracy (tokens often accepted)
  3. The target model's forward pass time dominates

  ## How It Works

  1. **Draft phase**: Small model generates K tokens autoregressively
  2. **Verify phase**: Target model processes all K+1 positions in one pass
  3. **Accept/reject**: Compare draft vs target distributions, accept prefix
  4. **Resample**: Sample next token from adjusted distribution

  ## Acceptance Criteria

  Using rejection sampling to maintain exact target distribution:
  - Accept draft token if: r < min(1, p_target(x) / p_draft(x))
  - On rejection: sample from adjusted distribution

  ## Configuration

    * `:draft_model` - Path/repo for draft model
    * `:num_speculative_tokens` - Tokens to propose per iteration (default: 5)
    * `:acceptance_threshold` - Minimum acceptance probability (default: 0.0)
  """

  use GenServer
  require Logger

  defstruct [
    :draft_serving,
    :target_serving,
    :num_speculative_tokens,
    :acceptance_threshold,
    :stats
  ]

  # Client API

  @doc """
  Starts the speculative decoding module.

  ## Options

    * `:draft_model` - HuggingFace repo for draft model (required)
    * `:target_serving` - Name of target Nx.Serving (required)
    * `:num_speculative_tokens` - Tokens per speculation (default: 5)
    * `:acceptance_threshold` - Min acceptance prob (default: 0.0)
  """
  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Generates tokens using speculative decoding.

  Returns a stream of generated tokens.
  """
  def generate(prompt_tokens, opts \\ []) do
    GenServer.call(__MODULE__, {:generate, prompt_tokens, opts}, :infinity)
  end

  @doc """
  Returns speculative decoding statistics.
  """
  def stats do
    GenServer.call(__MODULE__, :stats)
  end

  @doc """
  Runs a single speculative iteration.

  Returns `{accepted_tokens, next_token}`.
  """
  def speculate(input_tokens, draft_logits_fn, target_logits_fn, opts \\ []) do
    num_tokens = Keyword.get(opts, :num_speculative_tokens, 5)
    temperature = Keyword.get(opts, :temperature, 1.0)

    # Generate draft tokens
    {draft_tokens, draft_probs} = generate_draft_tokens(input_tokens, draft_logits_fn, num_tokens, temperature)

    # Verify with target model
    {accepted, next_token, acceptance_rate} = verify_tokens(
      input_tokens,
      draft_tokens,
      draft_probs,
      target_logits_fn,
      temperature
    )

    {accepted, next_token, acceptance_rate}
  end

  # Server Callbacks

  @impl true
  def init(opts) do
    num_speculative_tokens = Keyword.get(opts, :num_speculative_tokens, 5)
    acceptance_threshold = Keyword.get(opts, :acceptance_threshold, 0.0)

    state = %__MODULE__{
      draft_serving: Keyword.get(opts, :draft_serving),
      target_serving: Keyword.get(opts, :target_serving),
      num_speculative_tokens: num_speculative_tokens,
      acceptance_threshold: acceptance_threshold,
      stats: %{
        total_iterations: 0,
        total_accepted: 0,
        total_proposed: 0,
        avg_acceptance_rate: 0.0
      }
    }

    Logger.info("Speculative: Started with num_tokens=#{num_speculative_tokens}")

    {:ok, state}
  end

  @impl true
  def handle_call({:generate, prompt_tokens, opts}, _from, state) do
    max_tokens = Keyword.get(opts, :max_tokens, 100)

    tokens =
      Stream.unfold({prompt_tokens, 0}, fn
        {_tokens, count} when count >= max_tokens ->
          nil

        {tokens, count} ->
          # For now, return a placeholder - real implementation would use draft/target models
          # This is the interface; actual model calls happen in the serving layer
          {new_token, new_count} = do_speculative_step(tokens, state, opts)
          {{new_token, count + new_count}, {tokens ++ [new_token], count + new_count}}
      end)
      |> Stream.map(fn {token, _} -> token end)

    {:reply, tokens, state}
  end

  @impl true
  def handle_call(:stats, _from, state) do
    {:reply, state.stats, state}
  end

  # Private functions

  defp generate_draft_tokens(input_tokens, draft_logits_fn, num_tokens, temperature) do
    {tokens, probs, _} =
      Enum.reduce(1..num_tokens, {[], [], input_tokens}, fn _i, {tokens, probs, context} ->
        logits = draft_logits_fn.(context)
        prob_dist = softmax_with_temperature(logits, temperature)
        token = sample_from_distribution(prob_dist)
        token_prob = Nx.to_number(prob_dist[token])

        {tokens ++ [token], probs ++ [token_prob], context ++ [token]}
      end)

    {tokens, probs}
  end

  defp verify_tokens(input_tokens, draft_tokens, draft_probs, target_logits_fn, temperature) do
    # Get target logits for all positions at once
    all_tokens = input_tokens ++ draft_tokens
    target_logits = target_logits_fn.(all_tokens)

    # Verify each draft token
    {accepted, rejection_idx} =
      draft_tokens
      |> Enum.with_index()
      |> Enum.reduce_while({[], nil}, fn {draft_token, idx}, {accepted, _} ->
        # Get target probability for this position
        pos = length(input_tokens) + idx
        target_prob_dist = softmax_with_temperature(target_logits[pos], temperature)
        target_prob = Nx.to_number(target_prob_dist[draft_token])
        draft_prob = Enum.at(draft_probs, idx)

        # Rejection sampling criterion
        acceptance_prob = min(1.0, target_prob / max(draft_prob, 1.0e-10))
        r = :rand.uniform()

        if r < acceptance_prob do
          {:cont, {accepted ++ [draft_token], nil}}
        else
          {:halt, {accepted, idx}}
        end
      end)

    # Sample next token from adjusted distribution
    next_token =
      case rejection_idx do
        nil ->
          # All accepted, sample from target at position after last draft
          pos = length(input_tokens) + length(draft_tokens)
          target_prob_dist = softmax_with_temperature(target_logits[pos], temperature)
          sample_from_distribution(target_prob_dist)

        idx ->
          # Rejected at idx, sample from adjusted distribution
          pos = length(input_tokens) + idx
          target_prob_dist = softmax_with_temperature(target_logits[pos], temperature)
          draft_prob = Enum.at(draft_probs, idx)

          # Adjusted distribution: max(0, p_target - p_draft) normalized
          adjusted = adjust_distribution(target_prob_dist, draft_prob)
          sample_from_distribution(adjusted)
      end

    acceptance_rate = length(accepted) / max(length(draft_tokens), 1)

    {accepted, next_token, acceptance_rate}
  end

  defp softmax_with_temperature(logits, temperature) when temperature > 0 do
    scaled = Nx.divide(logits, temperature)
    Nx.exp(Nx.subtract(scaled, Nx.reduce_max(scaled)))
    |> then(fn exp_logits -> Nx.divide(exp_logits, Nx.sum(exp_logits)) end)
  end

  defp softmax_with_temperature(logits, _temperature) do
    # Temperature 0 = argmax (greedy)
    idx = Nx.argmax(logits) |> Nx.to_number()
    Nx.broadcast(Nx.tensor(0.0), Nx.shape(logits))
    |> Nx.put_slice([idx], Nx.tensor([1.0]))
  end

  defp sample_from_distribution(prob_dist) do
    # Multinomial sampling
    r = :rand.uniform()
    probs = Nx.to_flat_list(prob_dist)

    {_, idx} =
      Enum.reduce_while(probs, {0.0, 0}, fn p, {cumsum, idx} ->
        new_cumsum = cumsum + p
        if new_cumsum >= r do
          {:halt, {new_cumsum, idx}}
        else
          {:cont, {new_cumsum, idx + 1}}
        end
      end)

    idx
  end

  defp adjust_distribution(target_dist, draft_prob) do
    # max(0, p_target - p_draft), then normalize
    draft_tensor = Nx.broadcast(Nx.tensor(draft_prob), Nx.shape(target_dist))
    adjusted = Nx.max(Nx.subtract(target_dist, draft_tensor), 0)
    sum = Nx.sum(adjusted)

    if Nx.to_number(sum) > 0 do
      Nx.divide(adjusted, sum)
    else
      target_dist
    end
  end

  defp do_speculative_step(_tokens, _state, _opts) do
    # Placeholder - real implementation integrates with Nx.Serving
    # Returns {next_token, num_accepted}
    {0, 1}
  end
end
