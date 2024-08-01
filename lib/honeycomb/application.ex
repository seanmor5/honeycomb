defmodule Honeycomb.Application do
  use Application

  def start(_type, _args) do
    children = []

    children =
      if Application.fetch_env!(:honeycomb, :start_router) do
        router = {Bandit, plug: Honeycomb.Router}
        [router | children]
      else
        children
      end

    children =
      if Application.fetch_env!(:honeycomb, :start_serving) do
        serving =
          {Nx.Serving,
           serving: Honeycomb.Serving.serving(),
           name: Honeycomb.Serving,
           batch_size: 1,
           batch_timeout: 50}

        [serving | children]
      else
        children
      end

    opts = [strategy: :one_for_one, name: Honeycomb.Supervisor]
    Supervisor.start_link(children, opts)
  end
end
