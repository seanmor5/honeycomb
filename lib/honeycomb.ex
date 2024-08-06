defmodule Honeycomb do
  @moduledoc """
  Fast LLM inference built on Elixir, [Bumblebee](https://github.com/elixir-nx/bumblebee),
  and [EXLA](https://github.com/elixir-nx/nx/tree/main/exla).

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

    * `--max-sequence-length` - Text generation max sequence length. Total sequence
      length accounts for both input and output tokens.

    * `--hf-auth-token` - HuggingFace auth token for accessing private or gated repos.

  The Honeycomb server is compatible with the OpenAI API, so you can use it as a
  drop-in replacement by changing the `api_url` in the OpenAI client.

  ### As a dependency

  To use Honeycomb as a dependency, first add it to your `deps`:

      defp deps do
        [{:honeycomb, "~> 0.1"}]
      end

  Next, you'll need to configure the serving options:

      config :honeycomb, Honeycomb.Serving,
        model: "microsoft/Phi-3-mini-4k-instruct",
        chat_template: "phi3",
        auth_token: System.fetch_env!("HF_TOKEN")

  Then you can call Honeycomb directly:

      Honeycomb.chat_completion(...)
  """
  alias Honeycomb.OpenAI

  @doc """
  Generate a chat completion response.

  Note that currently many OpenAI options are not implemented. These options
  will still be validated, but ultimately ignored.

  ## Options

    * `:messages` - chat history as system/user/assistant prompts. Required

    * `:stream` - whether or not to stream the output. Defaults to `false`
  """
  def chat_completion(opts \\ []) do
    OpenAI.chat_completion(opts)
  end
end
