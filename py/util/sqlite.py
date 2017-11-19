# Copyright 2017 Bruno Ribeiro, Mayank Kakodkar, Pedro Savarese
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sqlite3 as lite

from util.config import SQLITE_FILE, LOCAL, TOUR_LENGTHS_TABLE, BASE_FOLDER, GPU_ID, GPU_LIMIT


class SQLite:
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

    def insert_tour_length_data(self, config, iter, hidden, k, epoch, total_tours, completed_tours, diverse_tours,
                                supernode_size,
                                supernode_samples, tour_lengths, mini_batch_id):
        self.execute_update(
            "CREATE TABLE IF NOT EXISTS " + TOUR_LENGTHS_TABLE + " ( \
              i INTEGER PRIMARY KEY AUTOINCREMENT \
              , time timestamp DEFAULT current_timestamp \
              , config TEXT \
              , iter TEXT \
              , hidden TEXT \
              , k TEXT \
              , epoch TEXT \
              , total_tours TEXT \
              , completed_tours TEXT \
              , diverse_tours TEXT \
              , supernode_size TEXT \
              , supernode_samples TEXT \
              , tour_lengths TEXT \
              , mini_batch_id TEXT \
              );"
        )
        self.execute_update(
            "INSERT INTO " + TOUR_LENGTHS_TABLE + "(config, iter, hidden, k, epoch, total_tours, completed_tours, diverse_tours, supernode_size, supernode_samples, tour_lengths, mini_batch_id) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            [config, iter, hidden, k, epoch, total_tours, completed_tours, diverse_tours, supernode_size,
             supernode_samples, tour_lengths, mini_batch_id])

    def insert_final_log(self, model_location, Z, L_train, L_test, name, fitted, iteration, iterations):
        method = fitted.method
        cdk = fitted.cdk
        mclvk = fitted.mclvk
        wm = fitted.warmup_epochs
        tot = fitted.max_epochs
        mbs = fitted.batch_size
        learning_rate = fitted.learning_rate
        weight_decay = fitted.weight_decay
        momentum = fitted.momentum
        plateau = fitted.plateau
        hidden = fitted.num_hidden
        supernode_samples = fitted.supernode_samples
        gpu_id = GPU_ID
        gpu_limit = GPU_LIMIT
        basefolder = BASE_FOLDER
        local = LOCAL

        self.execute_update(

            "CREATE TABLE IF NOT EXISTS FINAL_LIKELIHOODS_TABLE ( \
                i INTEGER PRIMARY KEY AUTOINCREMENT \
                , time timestamp DEFAULT current_timestamp \
                , local TEXT \
                , model_location TEXT \
                , basefolder TEXT \
                , method TEXT \
                , cdk TEXT \
                , mclvk TEXT \
                , wm TEXT \
                , tot TEXT \
                , mbs TEXT \
                , learning_rate TEXT \
                , weight_decay TEXT \
                , momentum TEXT \
                , plateau TEXT \
                , hidden TEXT \
                , supernode_samples TEXT \
                , gpu_id TEXT \
                , gpu_limit TEXT \
                , Z TEXT \
                , L_train TEXT \
                , L_test TEXT \
                , name TEXT \
                , iteration TEXT \
                , iterations TEXT \
              );"
        )
        self.execute_update(
            "INSERT INTO FINAL_LIKELIHOODS_TABLE(local \
            , model_location \
            , basefolder \
            , method \
            , cdk \
            , mclvk \
            , wm \
            , tot \
            , mbs \
            , learning_rate \
            , weight_decay \
            , momentum \
            , plateau \
            , hidden \
            , supernode_samples \
            , gpu_id \
            , gpu_limit \
            , Z \
            , L_train \
            , L_test \
            , name \
            , iteration \
            , iterations \
             ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [local
                , model_location
                , basefolder
                , str(method)
                , cdk
                , mclvk
                , wm
                , tot
                , mbs
                , learning_rate
                , weight_decay
                , momentum
                , plateau
                , hidden
                , supernode_samples
                , gpu_id
                , gpu_limit
                , Z
                , L_train
                , L_test
                , str(name)
                , iteration
                , iterations
             ])

    def execute_update(self, sql, parameters=()):
        connection = self.get_connection()
        connection.execute(sql, parameters)
        connection.commit()
        connection.close()

    def get_tour_data(self):
        return self.execute_query("SELECT * FROM " + TOUR_LENGTHS_TABLE)
