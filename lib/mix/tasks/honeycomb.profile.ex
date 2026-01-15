defmodule Mix.Tasks.Honeycomb.Profile do
  @moduledoc """
  Profiles the given Honeycomb configuration using fprof.

  ## Usage

      mix honeycomb.profile --model=microsoft/Phi-3-mini-4k-instruct --chat-template=phi3

  ## Options

  #{Honeycomb.CLI.usage()}
  """

  use Mix.Task

  @shortdoc "Profiles the given Honeycomb configuration"

  @prompt "Complete the following: The quick brown"

  @impl true
  def run(args) do
    case Honeycomb.CLI.parse_args(args) do
      {:ok, config} ->
        run_profile(config)

      {:help, usage} ->
        Mix.shell().info(usage)

      {:error, reason} ->
        Mix.shell().error("Error: #{reason}")
        Mix.shell().info(Honeycomb.CLI.usage())
        exit({:shutdown, 1})
    end
  end

  defp run_profile(config) do
    # Apply configuration
    :ok = Honeycomb.CLI.apply_config(config)

    Application.put_env(:honeycomb, :start_serving, true)

    Mix.Task.run("app.start")

    Mix.shell().info("Profiling chat completion...")
    Mix.shell().info("Model: #{config.model}")
    Mix.shell().info("Prompt: #{@prompt}")
    Mix.shell().info("")

    messages = [%{role: "user", content: @prompt}]
    opts = [messages: messages, stream: false]

    Mix.Tasks.Profile.Fprof.profile(
      fn ->
        Honeycomb.chat_completion(opts)
      end,
      details: true,
      callers: true
    )
  end
end
