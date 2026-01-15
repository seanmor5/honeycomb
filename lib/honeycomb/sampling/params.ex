defmodule Honeycomb.Sampling.Params do
  @moduledoc """
  Sampling parameters for token generation.

  Configures the various sampling strategies and penalties.
  """

  defstruct [
    # Temperature scaling
    temperature: 1.0,

    # Filtering strategies
    top_k: 0,
    top_p: 1.0,
    min_p: 0.0,

    # Repetition control
    repetition_penalty: 1.0,
    frequency_penalty: 0.0,
    presence_penalty: 0.0,

    # Context for penalties
    context_tokens: [],
    token_counts: %{},

    # Reproducibility
    seed: nil,

    # Stop conditions
    stop_sequences: [],
    max_tokens: nil,

    # Advanced
    typical_p: 1.0,
    mirostat_mode: 0,
    mirostat_tau: 5.0,
    mirostat_eta: 0.1
  ]

  @doc """
  Creates new sampling parameters.

  ## Options

    * `:temperature` - Sampling temperature (default: 1.0)
    * `:top_k` - Top-K filtering, 0 to disable (default: 0)
    * `:top_p` - Nucleus sampling threshold (default: 1.0)
    * `:min_p` - Min-P filtering threshold (default: 0.0)
    * `:repetition_penalty` - Penalty for repeated tokens (default: 1.0)
    * `:frequency_penalty` - OpenAI-style frequency penalty (default: 0.0)
    * `:presence_penalty` - OpenAI-style presence penalty (default: 0.0)
    * `:seed` - Random seed for reproducibility (default: nil)
    * `:stop_sequences` - Token sequences that stop generation (default: [])
    * `:max_tokens` - Maximum tokens to generate (default: nil)
  """
  def new(opts \\ []) do
    struct(__MODULE__, opts)
  end

  @doc """
  Creates parameters from OpenAI API options.
  """
  def from_openai(opts) when is_list(opts) do
    %__MODULE__{
      temperature: Keyword.get(opts, :temperature, 1.0),
      top_p: Keyword.get(opts, :top_p, 1.0),
      frequency_penalty: Keyword.get(opts, :frequency_penalty, 0.0),
      presence_penalty: Keyword.get(opts, :presence_penalty, 0.0),
      max_tokens: Keyword.get(opts, :max_tokens),
      seed: Keyword.get(opts, :seed),
      stop_sequences: parse_stop_sequences(Keyword.get(opts, :stop, []))
    }
  end

  @doc """
  Updates parameters with new context tokens.
  """
  def with_context(%__MODULE__{} = params, tokens) when is_list(tokens) do
    token_counts =
      Enum.reduce(tokens, %{}, fn token, acc ->
        Map.update(acc, token, 1, &(&1 + 1))
      end)

    %{params | context_tokens: tokens, token_counts: token_counts}
  end

  @doc """
  Adds a generated token to the context.
  """
  def add_token(%__MODULE__{} = params, token) do
    context_tokens = params.context_tokens ++ [token]
    token_counts = Map.update(params.token_counts, token, 1, &(&1 + 1))
    %{params | context_tokens: context_tokens, token_counts: token_counts}
  end

  @doc """
  Checks if generation should stop based on stop sequences.
  """
  def should_stop?(%__MODULE__{stop_sequences: []}, _tokens), do: false

  def should_stop?(%__MODULE__{stop_sequences: sequences}, tokens) do
    Enum.any?(sequences, fn seq ->
      ends_with_sequence?(tokens, seq)
    end)
  end

  @doc """
  Validates the parameters.
  """
  def validate(%__MODULE__{} = params) do
    cond do
      params.temperature < 0 ->
        {:error, "temperature must be non-negative"}

      params.top_k < 0 ->
        {:error, "top_k must be non-negative"}

      params.top_p < 0 or params.top_p > 1 ->
        {:error, "top_p must be between 0 and 1"}

      params.min_p < 0 or params.min_p > 1 ->
        {:error, "min_p must be between 0 and 1"}

      params.repetition_penalty < 0 ->
        {:error, "repetition_penalty must be non-negative"}

      params.frequency_penalty < -2 or params.frequency_penalty > 2 ->
        {:error, "frequency_penalty must be between -2 and 2"}

      params.presence_penalty < -2 or params.presence_penalty > 2 ->
        {:error, "presence_penalty must be between -2 and 2"}

      true ->
        :ok
    end
  end

  # Private helpers

  defp parse_stop_sequences(nil), do: []
  defp parse_stop_sequences(seq) when is_binary(seq), do: [seq]
  defp parse_stop_sequences(seqs) when is_list(seqs), do: seqs

  defp ends_with_sequence?(tokens, sequence) when is_binary(sequence) do
    # For string sequences, would need tokenizer - placeholder
    false
  end

  defp ends_with_sequence?(tokens, sequence) when is_list(sequence) do
    len = length(sequence)
    token_len = length(tokens)

    if token_len >= len do
      Enum.take(tokens, -len) == sequence
    else
      false
    end
  end
end
