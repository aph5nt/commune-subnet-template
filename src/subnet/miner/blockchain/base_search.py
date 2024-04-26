class BaseGraphSearch:
    def execute_query(self, query: str):
        """Execute a query and return the result."""

    def execute_cypher_query(self, cypher_query: str):
        """Execute a cypher query and return the result."""

    def close(self):
        """Close the connection to the graph database."""


class BaseBalanceSearch:
    def execute_query(self, query: str):
        """Execute a query and return the result."""
