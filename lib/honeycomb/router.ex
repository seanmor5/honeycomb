defmodule Honeycomb.Router do
  use Plug.Router

  alias Honeycomb.Controller

  plug :match

  plug Plug.Parsers,
    parsers: [:json],
    json_decoder: Jason

  plug :dispatch

  post "/v1/chat/completions" do
    Controller.OpenAI.chat_completion(conn)
  end
end
