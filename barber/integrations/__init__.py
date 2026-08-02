"""Adapters that map a framework's document type onto barber's selection call.

Nothing in here is imported by ``import barber``. Each submodule pulls in the
framework it adapts, and `dependencies = []` is a promise the package keeps, so
paying for langchain-core has to be the caller's choice rather than a side
effect of importing barber. Import the adapter you want, directly:

    from barber.integrations.langchain import BarberDocumentCompressor
    from barber.integrations.llama_index import BarberNodePostprocessor

Install the framework alongside barber with the matching extra:

    pip install "barber-llm[langchain]"
    pip install "barber-llm[llamaindex]"

Every adapter here is exactly that — an adapter. It maps the framework's types
onto ``barber.trim`` and maps the result back. No scoring, thresholds, or
defaults are redefined; changing selection behaviour means changing
``SelectionConfig``, the same knob the rest of the library uses.
"""
