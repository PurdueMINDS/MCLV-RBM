import sqlite3 as lite

import pandas as pd

from u.config import SQLITE_FILE, TOUR_LENGTHS_TABLE, GPU_LIMIT, GPU_HARD_LIMIT, BASE_FOLDER, LOCAL, HOSTNAME, \
    NO_DB, AVAILABLE_GPUS, DISCRETE, NP_DT, USE_DOUBLE
from u.log import Log
from u.utils import Util
from u.version import VERSION


class SQLite:
    _table_verified = {}

    def __init__(self, db_file=SQLITE_FILE):
        self.db_file = db_file

    def get_connection(self):
        connection = lite.connect(self.db_file)
        connection.row_factory = lite.Row
        return connection

    def execute_query(self, sql):
        connection = self.get_connection()
        with connection:
            cursor = connection.cursor()
            cursor.execute(sql)
            data = cursor.fetchall()
            return data

    def execute_update(self, sql, parameters=(), first_attempt=True):
        if not NO_DB:
            try:
                with self.get_connection() as connection:
                    connection.execute(sql, parameters)
                    connection.commit()
            except Exception as e:
                if first_attempt:
                    Log.exception("Retrying Insert", stack_info=True)
                    self.execute_update(sql, parameters=parameters, first_attempt=False)
                else:
                    Log.exception("Insert failed", stack_info=True)
                    if LOCAL:
                        raise e

    @staticmethod
    def insert_dict(table, log_dict):
        log_dict["gpu_id"] = AVAILABLE_GPUS
        log_dict["gpu_limit"] = GPU_LIMIT
        log_dict["gpu_hard_limit"] = GPU_HARD_LIMIT
        log_dict["basefolder"] = BASE_FOLDER
        log_dict["local"] = LOCAL
        log_dict["hostname"] = HOSTNAME
        log_dict["version"] = VERSION
        log_dict["discrete"] = DISCRETE
        log_dict["double"] = USE_DOUBLE

        keys = list(log_dict.keys())

        SQLite.verify_table(table, keys)

        start = "INSERT INTO " + table + "("
        mid_1 = ", ".join(keys)
        mid_2 = ") VALUES("
        mid_3 = ", ".join(["?" for _ in keys])
        end = ");"

        SQLite().execute_update(start + mid_1 + mid_2 + mid_3 + end, [str(log_dict[s]) for s in keys])

    @staticmethod
    def verify_table(table, keys):
        if table not in SQLite._table_verified:
            exists = pd.read_sql("SELECT name \
              FROM sqlite_master \
              WHERE type = 'table' AND name = '%s';" % table
                                 , SQLite().get_connection())
            if len(exists) == 0:
                su = "CREATE TABLE " + table + " (i INTEGER PRIMARY KEY AUTOINCREMENT , time timestamp DEFAULT current_timestamp, "
                end = ");"
                mid = ", ".join([s + " TEXT" for s in keys])
                SQLite().execute_update(su + mid + end)
                Log.info("Created new table %s", table)
            else:
                col_info = pd.read_sql("PRAGMA table_info(%s);" % table
                                       , SQLite().get_connection())
                existing_columns = set(col_info['name'])
                to_add = [s for s in keys if s not in existing_columns]
                if Util.empty(to_add):
                    Log.info("Verified table %s", table)
                else:
                    for col in to_add:
                        SQLite().execute_update("ALTER TABLE %s ADD %s TEXT" % (table, col))
                    Log.info("Modified table %s", table)

            SQLite._table_verified[table] = True
        return True