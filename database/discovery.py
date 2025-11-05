import psycopg2
import json
from datetime import datetime

class DatabaseDiscoveryService:
    """
    A service to discover the schema of a PostgreSQL database,
    including tables, columns, primary keys, and foreign key relationships.
    """
    def __init__(self, db_params):
        """
        Initializes the service with database connection parameters.
        :param db_params: A dictionary with dbname, user, password, host, port.
        """
        self.db_params = db_params
        self.connection = None

    def __enter__(self):
        """Opens the database connection."""
        self.connection = psycopg2.connect(**self.db_params)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Closes the database connection."""
        if self.connection:
            self.connection.close()

    def _execute_query(self, query):
        """Executes a query and returns the results."""
        with self.connection.cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchall()

    def discover_schema(self, force_refresh=False):
        """
        Discovers the full database schema, using a cache if available.
        :param force_refresh: If True, ignores the cache and re-discovers.
        """
        cache_path = "database/schema_cache.json"
        if not force_refresh:
            try:
                with open(cache_path, 'r') as f:
                    print("Loading schema from cache.")
                    return json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                print("Cache not found or invalid. Discovering schema...")

        print("Starting database schema discovery...")
        schema_map = {
            "tables": {},
            "relationships": []
        }

        with self as discoverer:
            tables = discoverer._get_tables()
            for schema, table_name in tables:
                if schema not in schema_map["tables"]:
                    schema_map["tables"][schema] = {}

                columns = discoverer._get_columns(schema, table_name)
                pk = discoverer._get_primary_key(schema, table_name)

                schema_map["tables"][schema][table_name] = {
                    "columns": columns,
                    "primary_key": pk
                }

            schema_map["relationships"] = discoverer._get_foreign_keys()

        # Save to cache
        with open(cache_path, 'w') as f:
            json.dump(schema_map, f, indent=4)
        print("Schema discovery completed and saved to cache.")
        return schema_map

    def _get_tables(self):
        """Retrieves all tables from the public schema."""
        query = """
        SELECT table_schema, table_name
        FROM information_schema.tables
        WHERE table_schema NOT IN ('pg_catalog', 'information_schema');
        """
        return self._execute_query(query)

    def _get_columns(self, schema_name, table_name):
        """Retrieves columns for a specific table."""
        query = f"""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = '{schema_name}' AND table_name = '{table_name}';
        """
        return {col: data_type for col, data_type in self._execute_query(query)}

    def _get_primary_key(self, schema_name, table_name):
        """Retrieves the primary key for a specific table."""
        query = f"""
        SELECT c.column_name
        FROM information_schema.key_column_usage AS c
        LEFT JOIN information_schema.table_constraints AS t
        ON t.constraint_name = c.constraint_name
        WHERE t.table_schema = '{schema_name}' AND t.table_name = '{table_name}' AND t.constraint_type = 'PRIMARY KEY';
        """
        result = self._execute_query(query)
        return result[0][0] if result else None

    def _get_foreign_keys(self):
        """Retrieves all foreign key relationships in the database."""
        query = """
        SELECT
            kcu.table_schema AS fk_schema,
            kcu.table_name AS fk_table,
            kcu.column_name AS fk_column,
            ccu.table_schema AS pk_schema,
            ccu.table_name AS pk_table,
            ccu.column_name AS pk_column
        FROM
            information_schema.key_column_usage AS kcu
        JOIN
            information_schema.table_constraints AS tc ON tc.constraint_name = kcu.constraint_name
        JOIN
            information_schema.constraint_column_usage AS ccu ON ccu.constraint_name = tc.constraint_name
        WHERE
            tc.constraint_type = 'FOREIGN KEY';
        """
        relationships = []
        for fk_schema, fk_table, fk_column, pk_schema, pk_table, pk_column in self._execute_query(query):
            relationships.append({
                "from_table": f"{fk_schema}.{fk_table}",
                "from_column": fk_column,
                "to_table": f"{pk_schema}.{pk_table}",
                "to_column": pk_column,
            })
        return relationships
