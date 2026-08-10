"""msql: the sqlite + msgpack middleware of ManicDrummer (manic.sqlite). Writer side lives in dio.build;
readers (model / eval / tools) come in through handler.connect_ro so the corpus can never be mutated."""
from .handler import connect_ro, connect_rw, read_ir_grid, read_ir_rich, read_meta, write_charts
from .schema import COLUMNS, SCHEMA
