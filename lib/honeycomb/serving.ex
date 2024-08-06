defmodule Honeycomb.Serving do
  @moduledoc false

  require Logger

  alias Honeycomb.Templates

  @default_sequence_length 512

  def serving() do
    model = env(:model)
    template = env(:chat_template)
    sequence_length = env(:sequence_length) || @default_sequence_length

    Logger.info("Serving: Using model repo #{model}")
    Logger.info("Serving: Using chat template #{template}")

    repo = repo()

    {:ok, model_info} = Bumblebee.load_model(repo)
    {:ok, tokenizer} = Bumblebee.load_tokenizer(repo)
    {:ok, generation_config} = Bumblebee.load_generation_config(repo)

    Bumblebee.Text.generation(model_info, tokenizer, generation_config,
      compile: [batch_size: 1, sequence_length: sequence_length],
      defn_options: [compiler: EXLA],
      stream: true,
      stream_done: true
    )
  end

  def model() do
    env(:model)
  end

  def generate(messages) do
    template = env(:chat_template)
    prompt = Templates.apply_chat_template(template, messages)
    Nx.Serving.batched_run(__MODULE__, prompt)
  end

  defp repo() do
    case env(:auth_token) do
      nil -> {:hf, env(:model)}
      token -> {:hf, env(:model), auth_token: token}
    end
  end

  defp env(key), do: Application.fetch_env!(:honeycomb, __MODULE__)[key]
end
