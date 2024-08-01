import Config

config :honeycomb, :start_serving, true
config :honeycomb, :start_router, false

config :honeycomb, Honeycomb.Serving,
  model: "microsoft/Phi-3-mini-4k-instruct",
  chat_template: "phi3"

config :nx, default_backend: EXLA.Backend
