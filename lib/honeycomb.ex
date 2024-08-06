defmodule Honeycomb do
  @moduledoc """
  Fast LLM inference built on Elixir and Bumblebee.

  ## Usage

  Honeycomb can be used as a standalone inference service or as a
  dependency in an existing Elixir project.

  ### As a separate service

  To use Honeycomb as a separate service, you just need to clone the project
  and run:


      mix honeycomb.serve <config>


  The following arguments are required:

    * `--model` - HuggingFace model repo to use
    * `--chat-template` - Chat template to use

  The following arguments are optional:

    * `--auth-token` - HuggingFace auth token for accessing private or gated repos.

  ### As a dependency

  To use Honeycomb as a dependency, first add it to your `deps`:

      defp deps do
        [{:honeycomb, "~> 0.1"}]
      end

  Next, you'll need to configure the serving options:

      config :honeycomb, Honeycomb.Serving,
        model: "google/gemma-2-2b-it",
        chat_template: "gemma2",
        auth_token: System.fetch_env!("HF_TOKEN")

  Then you can call Honeycomb directly:

      Honeycomb.chat_completion(...)
  """
  alias Honeycomb.OpenAI

  @doc """
  Generate a chat completion response.
  """
  def chat_completion(opts \\ []) do
    OpenAI.chat_completion(opts)
  end
end
