defmodule Honeycomb.MixProject do
  use Mix.Project

  @version "0.2.0"
  @source_url "https://github.com/seanmor5/honeycomb"

  def project do
    [
      app: :honeycomb,
      version: @version,
      elixir: "~> 1.16",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      name: "Honeycomb",
      description: "Production-ready LLM inference server for Elixir",
      source_url: @source_url,
      docs: docs(),
      package: package()
    ]
  end

  def application do
    [
      mod: {Honeycomb.Application, []},
      extra_applications: [:logger, :crypto]
    ]
  end

  defp deps do
    [
      # HTTP server
      {:bandit, "~> 1.0"},

      # JSON encoding
      {:jason, "~> 1.4"},

      # ML stack
      {:bumblebee, github: "elixir-nx/bumblebee"},
      {:exla, ">= 0.0.0"},

      # Validation
      {:nimble_options, "~> 1.0"},

      # Telemetry & observability
      {:telemetry, "~> 1.0"},

      # Documentation
      {:ex_doc, "~> 0.30", only: :dev, runtime: false}
    ]
  end

  defp docs do
    [
      main: "Honeycomb",
      extras: ["README.md"],
      groups_for_modules: [
        "Core": [
          Honeycomb,
          Honeycomb.Serving,
          Honeycomb.Router,
          Honeycomb.Templates
        ],
        "Engine": [
          Honeycomb.Engine,
          Honeycomb.Scheduler,
          Honeycomb.Scheduler.Request,
          Honeycomb.Scheduler.SchedulerConfig,
          Honeycomb.Scheduler.SchedulerOutput
        ],
        "Memory Management": [
          Honeycomb.KVCache,
          Honeycomb.KVCache.Block,
          Honeycomb.KVCache.BlockAllocator,
          Honeycomb.KVCache.BlockTable,
          Honeycomb.MemoryPool,
          Honeycomb.PrefixCache,
          Honeycomb.PrefixCache.RadixTree
        ],
        "Optimization": [
          Honeycomb.ChunkedPrefill,
          Honeycomb.Speculative,
          Honeycomb.Quantization,
          Honeycomb.TensorParallel
        ],
        "Sampling": [
          Honeycomb.Sampling,
          Honeycomb.Sampling.Params,
          Honeycomb.Logprobs
        ],
        "Observability": [
          Honeycomb.Telemetry,
          Honeycomb.Metrics
        ],
        "CLI": [
          Honeycomb.CLI
        ]
      ]
    ]
  end

  defp package do
    [
      licenses: ["MIT"],
      links: %{"GitHub" => @source_url}
    ]
  end
end
