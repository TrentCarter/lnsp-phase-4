"""
Provider Registry - SQLite backend for managing provider registrations

This module handles storage and retrieval of AI provider information including
capabilities, pricing, and performance SLOs.
"""

import sqlite3
import json
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path


class ProviderRegistry:
    """SQLite-based registry for AI provider management"""

    def __init__(self, db_path: str = "artifacts/provider_router/providers.db"):
        """Initialize provider registry with SQLite database"""
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Create providers table if it doesn't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS providers (
                name TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                context_window INTEGER NOT NULL,
                cost_per_input_token REAL NOT NULL,
                cost_per_output_token REAL NOT NULL,
                endpoint TEXT NOT NULL,
                features TEXT,
                slo TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                status TEXT DEFAULT 'active'
            )
        """)

        conn.commit()
        conn.close()

    def register_provider(self, provider_data: Dict) -> Dict:
        """
        Register a new provider or update existing one

        Args:
            provider_data: Provider registration data matching schema

        Returns:
            Dict with registration status and provider info
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        now = datetime.utcnow().isoformat()

        try:
            # Check if provider already exists
            cursor.execute("SELECT name FROM providers WHERE name = ?", (provider_data['name'],))
            exists = cursor.fetchone() is not None

            if exists:
                # Update existing provider
                cursor.execute("""
                    UPDATE providers
                    SET model = ?, context_window = ?, cost_per_input_token = ?,
                        cost_per_output_token = ?, endpoint = ?, features = ?,
                        slo = ?, metadata = ?, updated_at = ?
                    WHERE name = ?
                """, (
                    provider_data['model'],
                    provider_data['context_window'],
                    provider_data['cost_per_input_token'],
                    provider_data['cost_per_output_token'],
                    provider_data['endpoint'],
                    json.dumps(provider_data.get('features', [])),
                    json.dumps(provider_data.get('slo', {})),
                    json.dumps(provider_data.get('metadata', {})),
                    now,
                    provider_data['name']
                ))
                action = "updated"
            else:
                # Insert new provider
                cursor.execute("""
                    INSERT INTO providers (
                        name, model, context_window, cost_per_input_token,
                        cost_per_output_token, endpoint, features, slo,
                        metadata, created_at, updated_at, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active')
                """, (
                    provider_data['name'],
                    provider_data['model'],
                    provider_data['context_window'],
                    provider_data['cost_per_input_token'],
                    provider_data['cost_per_output_token'],
                    provider_data['endpoint'],
                    json.dumps(provider_data.get('features', [])),
                    json.dumps(provider_data.get('slo', {})),
                    json.dumps(provider_data.get('metadata', {})),
                    now,
                    now
                ))
                action = "registered"

            conn.commit()

            return {
                "status": "success",
                "action": action,
                "provider": provider_data['name'],
                "timestamp": now
            }

        except Exception as e:
            conn.rollback()
            return {
                "status": "error",
                "error": str(e),
                "timestamp": now
            }
        finally:
            conn.close()

    def get_provider(self, name: str) -> Optional[Dict]:
        """Get provider by name"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM providers WHERE name = ? AND status = 'active'", (name,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return self._row_to_dict(row)
        return None

    def list_providers(self, model: Optional[str] = None, min_context: Optional[int] = None) -> List[Dict]:
        """
        List all active providers with optional filtering

        Args:
            model: Filter by model name (exact match)
            min_context: Filter by minimum context window

        Returns:
            List of provider dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT * FROM providers WHERE status = 'active'"
        params = []

        if model:
            query += " AND model = ?"
            params.append(model)

        if min_context:
            query += " AND context_window >= ?"
            params.append(min_context)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_dict(row) for row in rows]

    def find_matching_providers(self, requirements: Dict) -> List[Dict]:
        """
        Find providers matching specific requirements

        Args:
            requirements: Dict with model, context_window, features

        Returns:
            List of matching providers sorted by cost (cheapest first)
        """
        providers = self.list_providers(
            model=requirements.get('model'),
            min_context=requirements.get('context_window')
        )

        # Filter by features if specified
        features_list = requirements.get('features') or []
        required_features = set(features_list) if features_list else set()
        if required_features:
            providers = [
                p for p in providers
                if required_features.issubset(set(p['features']))
            ]

        # Sort by total cost (input + output tokens)
        # Use average cost as proxy: (input_cost + output_cost) / 2
        providers.sort(key=lambda p: (p['cost_per_input_token'] + p['cost_per_output_token']) / 2)

        return providers

    def deactivate_provider(self, name: str) -> Dict:
        """Deactivate a provider (soft delete)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("UPDATE providers SET status = 'inactive' WHERE name = ?", (name,))
        conn.commit()
        rows_affected = cursor.rowcount
        conn.close()

        if rows_affected > 0:
            return {"status": "success", "action": "deactivated", "provider": name}
        else:
            return {"status": "error", "error": "Provider not found"}

    def _row_to_dict(self, row: sqlite3.Row) -> Dict:
        """Convert SQLite row to dictionary"""
        return {
            "name": row['name'],
            "model": row['model'],
            "context_window": row['context_window'],
            "cost_per_input_token": row['cost_per_input_token'],
            "cost_per_output_token": row['cost_per_output_token'],
            "endpoint": row['endpoint'],
            "features": json.loads(row['features']) if row['features'] else [],
            "slo": json.loads(row['slo']) if row['slo'] else {},
            "metadata": json.loads(row['metadata']) if row['metadata'] else {},
            "created_at": row['created_at'],
            "updated_at": row['updated_at'],
            "status": row['status']
        }

    def get_stats(self) -> Dict:
        """Get provider registry statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM providers WHERE status = 'active'")
        active_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM providers WHERE status = 'inactive'")
        inactive_count = cursor.fetchone()[0]

        cursor.execute("SELECT DISTINCT model FROM providers WHERE status = 'active'")
        unique_models = [row[0] for row in cursor.fetchall()]

        conn.close()

        return {
            "active_providers": active_count,
            "inactive_providers": inactive_count,
            "unique_models": unique_models,
            "total_models": len(unique_models)
        }
