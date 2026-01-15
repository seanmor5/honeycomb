defmodule Mix.Tasks.Honeycomb.Serve do
  @moduledoc """
  Starts the Honeycomb LLM inference server.

  ## Usage

      mix honeycomb.serve --model=microsoft/Phi-3-mini-4k-instruct --chat-template=phi3

  ## Options

  #{Honeycomb.CLI.usage()}
  """

  use Mix.Task

  @shortdoc "Starts the Honeycomb server"

  @impl true
  def run(args) do
    case Honeycomb.CLI.parse_args(args) do
      {:ok, config} ->
        start_server(config)

      {:help, usage} ->
        Mix.shell().info(usage)

      {:error, reason} ->
        Mix.shell().error("Error: #{reason}")
        Mix.shell().info(Honeycomb.CLI.usage())
        exit({:shutdown, 1})
    end
  end

  defp start_server(config) do
    # Apply configuration
    :ok = Honeycomb.CLI.apply_config(config)

    # Configure startup flags
    Application.put_env(:honeycomb, :start_serving, true)
    Application.put_env(:honeycomb, :start_router, true)
    Application.put_env(:honeycomb, :port, config.port)

    # Enable production engine if any advanced features are enabled
    enable_engine =
      config.tensor_parallel > 1 or
      config.enable_prefix_caching or
      config.enable_chunked_prefill or
      config.batch_size > 1

    Application.put_env(:honeycomb, :start_engine, enable_engine)

    Mix.shell().info("""
    Starting Honeycomb server...
      Model: #{config.model}
      Template: #{config.chat_template}
      Port: #{config.port}
      Batch size: #{config.batch_size}
      Quantization: #{config.quantization}
      Tensor parallel: #{config.tensor_parallel}
      Prefix caching: #{config.enable_prefix_caching}
      Chunked prefill: #{config.enable_chunked_prefill}
      Production engine: #{enable_engine}
    """)

    Mix.Tasks.Run.run(["--no-halt"])
  end
end
