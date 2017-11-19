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

import time


class Times:
    def __init__(self):
        self.start = time.time()
        self.times = []
        self.events = []

    def add(self, event):
        self.times.append(time.time())
        self.events.append(event)

    def compute(self):
        tmap = {}
        for time, event in zip(self.times, self.events):
            tmap[event] = time - self.start
            self.start = time
        return tmap
