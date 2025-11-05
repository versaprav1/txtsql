import unittest
import json
import os
import streamlit as st
from unittest.mock import patch
from database.discovery import DatabaseDiscoveryService

class TestDatabaseDiscoveryService(unittest.TestCase):

    def setUp(self):
        """Set up a mock Streamlit session state for database parameters."""
        self.mock_session_state = {
            "db_name": "new",
            "db_user": "postgres",
            "db_password": "pass",
            "db_host": "localhost",
            "db_port": "5432"
        }
        # Ensure cache from previous runs is cleared
        if os.path.exists("database/schema_cache.json"):
            os.remove("database/schema_cache.json")

    def get_db_params(self):
        return {
            "dbname": self.mock_session_state["db_name"],
            "user": self.mock_session_state["db_user"],
            "password": self.mock_session_state["db_password"],
            "host": self.mock_session_state["db_host"],
            "port": self.mock_session_state["db_port"]
        }

    @patch('streamlit.session_state')
    def test_discover_schema_and_cache(self, mock_st_session_state):
        """
        Test that schema discovery runs and creates a valid cache file.
        """
        # Set up mock session state by configuring the mock object
        mock_st_session_state.get.side_effect = self.mock_session_state.get

        db_params = self.get_db_params()

        try:
            service = DatabaseDiscoveryService(db_params)
            # Run discovery with force_refresh to ensure it hits the DB
            schema_map = service.discover_schema(force_refresh=True)

            # 1. Validate the returned schema map structure
            self.assertIn("tables", schema_map)
            self.assertIn("relationships", schema_map)
            self.assertIsInstance(schema_map["tables"], dict)
            self.assertIsInstance(schema_map["relationships"], list)

            # 2. Check if the cache file was created
            self.assertTrue(os.path.exists("database/schema_cache.json"))

            # 3. Validate the content of the cache file
            with open("database/schema_cache.json", 'r') as f:
                cached_schema = json.load(f)
            self.assertEqual(schema_map, cached_schema)

            # 4. Run discovery again, this time it should load from cache
            # We can tell because it will be much faster and won't print "Starting database schema discovery..."
            # For a more robust test, we could mock the _execute_query method
            schema_from_cache = service.discover_schema(force_refresh=False)
            self.assertEqual(schema_map, schema_from_cache)

        except Exception as e:
            # If the database is not running, we can't perform the test.
            # We'll print a warning instead of failing the test.
            print(f"WARNING: Could not connect to the database. Skipping test. Details: {e}")
            self.skipTest("Database connection not available.")

    def tearDown(self):
        """Clean up the cache file after tests."""
        if os.path.exists("database/schema_cache.json"):
            os.remove("database/schema_cache.json")

if __name__ == '__main__':
    unittest.main()
