# Honeycomb

Fast LLM inference built on Elixir, [Bumblebee](https://github.com/elixir-nx/bumblebee), and [EXLA](https://github.com/elixir-nx/nx/tree/main/exla).

## Usage

Honeycomb can be used as a standalone inference service or as a dependency in an existing Elixir project.

### As a separate service

To use Honeycomb as a separate service, you just need to clone the project and run:

```shell
mix honeycomb.serve <config>
```

The following arguments are required:

  * `--model` - HuggingFace model repo to use

  * `--chat-template` - Chat template to use

The following arguments are optional:

  * `--max-sequence-length` - Text generation max sequence length. Total sequence
    length accounts for both input and output tokens.

  * `--hf-auth-token` - HuggingFace auth token for accessing private or gated repos.

The Honeycomb server is compatible with the OpenAI API, so you can use it as a drop-in replacement by changing the `api_url` in the OpenAI client.

### As a dependency

To use Honeycomb as a dependency, first add it to your `deps`:
  
```elixir
defp deps do
  [{:honeycomb, github: "seanmor5/honeycomb"}]
end
```

Next, you'll need to configure the serving options:

```elixir
config :honeycomb, Honeycomb.Serving,
  model: "microsoft/Phi-3-mini-4k-instruct",
  chat_template: "phi3",
  auth_token: System.fetch_env!("HF_TOKEN")
```

Then you can call Honeycomb directly:

```elixir
messages = [%{role: "user", content: "Hello!"}]
Honeycomb.chat_completion(messages: messages)
```

## Benchmarks

Honeycomb ships with some basic benchmarks and profiling utilities. You can benchmark and/or profile your inference configuration by running:

```shell
mix honeycomb.benchmark <config>
```