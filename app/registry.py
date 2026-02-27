"""Shared app components initialized at startup."""


class AppRegistry:
    """Holds initialized app-level components.

    Attributes:
        graph: The compiled LangGraph agent graph.
    """

    graph = None


app_registry = AppRegistry()
