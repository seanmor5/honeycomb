defmodule Honeycomb.Logprobs do
  @moduledoc """
  Log probabilities computation for generated tokens.

  Provides OpenAI-compatible logprobs output including:
  - Log probability of each generated token
  - Top-K alternative tokens with their probabilities
  - Token byte offsets for string reconstruction

  ## Usage

      # During generation, compute logprobs
      logprobs = Honeycomb.Logprobs.compute(logits, token_id, opts)

      # Get top-K alternatives
      top_logprobs = Honeycomb.Logprobs.top_k(logits, k: 5)
  """

  import Nx.Defn

  @doc """
  Computes log probability information for a generated token.

  ## Options

    * `:top_logprobs` - Number of top tokens to include (default: nil)
    * `:tokenizer` - Tokenizer for decoding tokens (optional)
  """
  def compute(logits, token_id, opts \\ []) do
    k = Keyword.get(opts, :top_logprobs)
    tokenizer = Keyword.get(opts, :tokenizer)

    # Convert logits to log probabilities
    log_probs = log_softmax(logits)

    # Get log probability of the selected token
    token_logprob = Nx.to_number(log_probs[token_id])

    # Get token string if tokenizer available
    token_str = decode_token(tokenizer, token_id)

    result = %{
      token: token_str,
      token_id: token_id,
      logprob: token_logprob
    }

    # Add top-K if requested
    if k && k > 0 do
      top_tokens = top_k_logprobs(log_probs, k, tokenizer)
      Map.put(result, :top_logprobs, top_tokens)
    else
      result
    end
  end

  @doc """
  Computes top-K log probabilities from logits.
  """
  def top_k(logits, opts \\ []) do
    k = Keyword.get(opts, :k, 5)
    tokenizer = Keyword.get(opts, :tokenizer)

    log_probs = log_softmax(logits)
    top_k_logprobs(log_probs, k, tokenizer)
  end

  @doc """
  Builds a complete logprobs response object (OpenAI format).
  """
  def build_response(token_logprobs, text_offset \\ 0) do
    {tokens, logprobs, top_logprobs, offsets, _} =
      Enum.reduce(token_logprobs, {[], [], [], [], text_offset}, fn lp, {tokens, logprobs, top_lps, offsets, offset} ->
        token_str = lp[:token] || ""
        token_len = String.length(token_str)

        {
          tokens ++ [token_str],
          logprobs ++ [lp.logprob],
          top_lps ++ [lp[:top_logprobs]],
          offsets ++ [offset],
          offset + token_len
        }
      end)

    %{
      tokens: tokens,
      token_logprobs: logprobs,
      top_logprobs: if(Enum.any?(top_logprobs, & &1), do: top_logprobs, else: nil),
      text_offset: offsets
    }
  end

  @doc """
  Computes perplexity from a sequence of log probabilities.
  """
  def perplexity(log_probs) when is_list(log_probs) do
    n = length(log_probs)
    if n == 0 do
      0.0
    else
      avg_neg_log_prob = -Enum.sum(log_probs) / n
      :math.exp(avg_neg_log_prob)
    end
  end

  @doc """
  Computes entropy from logits.
  """
  defn entropy(logits) do
    probs = Nx.exp(log_softmax(logits))
    log_probs = log_softmax(logits)
    -Nx.sum(Nx.multiply(probs, log_probs))
  end

  @doc """
  Computes cross-entropy loss for a sequence.
  """
  def cross_entropy_loss(logits_sequence, target_tokens) do
    losses =
      Enum.zip(logits_sequence, target_tokens)
      |> Enum.map(fn {logits, target} ->
        log_probs = log_softmax(logits)
        -Nx.to_number(log_probs[target])
      end)

    Enum.sum(losses) / length(losses)
  end

  # Private helpers

  defnp log_softmax(logits) do
    max_logit = Nx.reduce_max(logits)
    shifted = Nx.subtract(logits, max_logit)
    log_sum_exp = Nx.log(Nx.sum(Nx.exp(shifted)))
    Nx.subtract(shifted, log_sum_exp)
  end

  defp top_k_logprobs(log_probs, k, tokenizer) do
    # Get indices of top-k values
    flat = Nx.to_flat_list(log_probs)

    flat
    |> Enum.with_index()
    |> Enum.sort_by(fn {prob, _idx} -> -prob end)
    |> Enum.take(k)
    |> Enum.map(fn {prob, idx} ->
      %{
        token: decode_token(tokenizer, idx),
        token_id: idx,
        logprob: prob
      }
    end)
  end

  defp decode_token(nil, token_id), do: "<token_#{token_id}>"

  defp decode_token(tokenizer, token_id) do
    try do
      Bumblebee.Tokenizer.decode(tokenizer, [token_id])
    rescue
      _ -> "<token_#{token_id}>"
    end
  end
end
