import Config

config :honeycomb, :start_serving, false
config :nx, default_backend: EXLA.Backend
